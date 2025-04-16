import os
from typing import Any, Callable, List, Union
from pathlib import Path
import numpy as np
import torch
import copy
import math
import random
from byprot import utils
from byprot.models.fixedbb.generator import IterativeRefinementGenerator
from byprot.modules import metrics
from byprot.tasks import TaskLitModule, register_task
from byprot.utils.config import compose_config as Cfg, merge_config

from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric
from Bio import pairwise2
from collections import Counter
import Levenshtein as lev

from byprot.datamodules.datasets.data_utils import Alphabet
from byprot.models.fixedbb.protein_mpnn_cmlm.protein_mpnn import ProteinMPNNCMLM
from byprot.models.fixedbb.pifold.pifold import PiFold
# import esm

log = utils.get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BLOSUM62_DATA = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0 -1  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2 -1 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -3
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -3  1 -1 -2 -2  0 -3 -1  4
"""

def parse_blosum62_matrix(matrix_str):
    lines = matrix_str.strip().split('\n')
    aa_list = lines[0].split()
    blosum62 = {aa: {} for aa in aa_list}
    for i, line in enumerate(lines[1:]):
        parts = line.split()
        aa = parts[0]
        scores = parts[1:]
        for j, score in enumerate(scores):
            blosum62[aa][aa_list[j]] = int(score)
    return blosum62

def stable_softmax(logits):
    # Shift logits to prevent overflow
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    exps = torch.exp(shifted_logits)
    sum_exps = torch.sum(exps, dim=-1, keepdim=True)
    return exps / sum_exps

class ProteinDesignPolicyNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super(ProteinDesignPolicyNetwork, self).__init__()
        self.policy_net = ProteinMPNNCMLM(cfg).to(device) # 直接初始化模型，不指定设备
        #self.policy_net = PiFold(cfg).to(device) # 直接初始化模型，不指定设备
    
        #proteinmpnn  
    def forward(self, batch):
        logits = self.policy_net(batch)
        logits = logits.float() 
        probs = F.softmax(logits, dim=-1)  # 将logits转换成概率
        return logits

    

    def sample_action(self, batch):
        """使用torch.no_grad减少内存消耗"""
        with torch.no_grad():
            probs = self.forward(batch)
            # 校正概率确保每行的和为1
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # 确保概率在[0, 1]范围内
            probs = torch.clamp(probs, 0, 1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            #print(f"Action shape: {action.shape}, Log prob shape: {log_prob.shape}")
            #print(f"Action: {action}, Log prob: {log_prob}")  # 打印动作和对数概率
        return action, log_prob



class REINFORCE:
    def __init__(self, cfg, learning_rate, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ProteinDesignPolicyNetwork(cfg).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rewards = []  # 初始化奖励列表
        self.saved_log_probs = []  # 初始化日志概率列表

    def take_action(self, batch):
        """使用策略网络采样行动并保存日志概率"""
        with torch.no_grad():
            action, log_prob = self.policy_net.sample_action(batch)
            self.saved_log_probs.append(log_prob)
            return action, log_prob
    
    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device, requires_grad=True)  # 确保返回值具有梯度
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for log_prob, R in zip(self.saved_log_probs, returns):
            # 确保计算中每一步都保持梯度
            loss = -log_prob * R
            if loss.requires_grad:
                policy_loss.append(loss.sum())
            else:
                # 如果发现没有梯度，打印出警告
                print("Warning: Loss computation without grad detected")

        if policy_loss:
            policy_loss = sum(policy_loss)  # 直接加总所有的损失
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
        else:
            print("No gradients to perform backpropagation on.")

        self.rewards.clear()
        self.saved_log_probs.clear()

    
    def store_reward(self, reward):
        """存储奖励"""
        self.rewards.append(reward)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('fixedbb/cmlm')
class CMLM(TaskLitModule):
    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='no_noise',  # ['full_mask', 'random_mask']
            num_unroll=0,
        ),
        generator=Cfg(
            max_iter=1,
            strategy='denoise',  # ['denoise' | 'mask_predict']
            noise='full_mask',  # ['full_mask' | 'selected mask']
            replace_visible_tokens=False,
            temperature=0,
            eval_sc=False,
        )
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        alphabet: DictConfig,
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
        generator=_DEFAULT_CFG.generator,
        reinforce=None 
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)

        self.alphabet = Alphabet(**alphabet)
        self.build_model() 
        self.build_generator()
        
        self.blosum62 = parse_blosum62_matrix(BLOSUM62_DATA)
        
        # 初始化目标权重
        self.weights = {
            'similarity': 0.8,
            'structure': 0.1,
            'diversity': 0.1
        }
        
        #self.policy_net = ProteinDesignPolicyNetwork(self.hparams.model).to(device)
        self.reinforce = REINFORCE(self.hparams.model, learning_rate=0.01, gamma=0.99)

        
        #self.reinforce = REINFORCE(
            #cfg=self.hparams.model,  
            #learning_rate=0.01,
            #gamma=0.99,
        #)
        
        optimizer_cfg = self.hparams.optimizer
        self.optimizer = AdamW(self.parameters(), lr=optimizer_cfg.lr, betas=optimizer_cfg.betas, weight_decay=optimizer_cfg.weight_decay)

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model').to(device)

    def build_generator(self):
        self.hparams.generator = merge_config(
            default_cfg=self._DEFAULT_CFG.generator,
            override_cfg=self.hparams.generator
        )
        self.generator = IterativeRefinementGenerator(
            alphabet=self.alphabet,
            **self.hparams.generator
        )
        log.info(f"Generator config: {self.hparams.generator}")

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.alphabet.padding_idx

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()

        self.acc = MeanMetric()
        self.acc_best = MaxMetric()

        self.acc_median = CatMetric()
        self.acc_median_best = MaxMetric()

    def load_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_epoch_start(self) -> None:
        if self.hparams.generator.eval_sc:
            import esm
            log.info(f"Eval structural self-consistency enabled. Loading ESMFold model...")
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            self._folding_model = self._folding_model.to(self.device)


    def smith_waterman_similarity(self, seq1, seq2):
        alignments = pairwise2.align.localms(seq1, seq2, 2, -1, -0.5, -0.1)
        if alignments:
            top_alignment = alignments[0]
            score = top_alignment.score
            max_possible_score = min(len(seq1), len(seq2)) * 2
            normalized_score = score / max_possible_score if max_possible_score > 0 else 0
        else:
            normalized_score = 0
        return normalized_score
    
    def sequence_similarity(self, pred_seqs, target_seq, method='smith_waterman'):
        target_seq = ''.join(target_seq) if isinstance(target_seq, list) else target_seq
        similarity_scores = []
        for pred_seq in pred_seqs:
            pred_seq_clean = ''.join(pred_seq).replace('&lt;pad&gt;', '')
            if method == 'smith_waterman':
                score = self.smith_waterman_similarity(pred_seq_clean, target_seq)
            else:
                score = self.calculate_similarity(pred_seq_clean, target_seq)  # Levenshtein或其他方法
            similarity_scores.append(score)
        average_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        return average_score

    def calculate_similarity(self,seq1, seq2):
        # 计算两个序列的Levenshtein距离
        distance = lev.distance(seq1, seq2)
        # 转换成相似度分数（越相似分数越高）
        max_len = max(len(seq1), len(seq2))
        similarity = (max_len - distance) / max_len if max_len != 0 else 0
        return similarity
    
    #相似性得分
    def sequence_similarity(self, pred_seqs, target_seq):
        # 确保target_seq是字符串
        target_seq = ''.join(target_seq) if isinstance(target_seq, list) else target_seq

        similarity_scores = []
        for pred_seq in pred_seqs:
            # 清洗预测序列，移除<pad>字符
            pred_seq_clean = ''.join(pred_seq).replace('<pad>', '') if isinstance(pred_seq, list) else pred_seq.replace('<pad>', '')
            # 计算相似度分数（此处假设有一个方法来计算两个序列的相似度）
            similarity_score = self.calculate_similarity(pred_seq_clean, target_seq)
            similarity_scores.append(similarity_score)
    
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    
    #计算的是每个氨基酸的二级结构偏好分数，并累加这些分数。移除填充字符后，仍然可以正确计算分数。偏好分数累可以帮助了解特定环境中的蛋白质可能倾向于形成哪种类型的二级结构
    def secondary_structure_preference(self, sequence):
        # 定义氨基酸在α螺旋和β折叠结构中的偏好分数
        helix_pref = {'A': 1.45, 'R': 0.79, 'N': 0.94, 'D': 0.98, 'C': 0.77, 'Q': 1.00, 'E': 1.53, 'G': 0.53, 'H': 1.24, 'I': 1.00, 'L': 1.34, 'K': 1.23, 'M': 1.20, 'F': 1.12, 'P': 0.59, 'S': 0.79, 'T': 0.82, 'W': 1.14,                       'Y': 0.61, 'V': 1.14}
        beta_pref = {'A': 0.97, 'R': 0.90, 'N': 0.65, 'D': 0.80, 'C': 1.30, 'Q': 1.23, 'E': 0.26, 'G': 0.81, 'H': 0.87, 'I': 1.60, 'L': 1.22, 'K': 0.74, 'M': 1.67, 'F': 1.28, 'P': 0.42, 'S': 0.72, 'T': 1.20, 'W': 1.19,                        'Y': 1.29, 'V': 1.65}

        # 计算非填充字符的α螺旋和β折叠偏好分数
        helix_score = sum([helix_pref.get(aa, 0) for aa in sequence if aa != '<pad>'])
        beta_score = sum([beta_pref.get(aa, 0) for aa in sequence if aa != '<pad>'])
    
        # 总分
        total_score = helix_score + beta_score
        return total_score
    
    #计算的是序列中不同氨基酸的种类数量与总长度的比值。移除填充字符后，仍然可以正确计算分数。
    def sequence_diversity(self, sequence):
        # 计算非填充字符的多样性分数
        counts = Counter([aa for aa in sequence if aa != '<pad>'])
        diversity_score = len(counts) / len([aa for aa in sequence if aa != '<pad>'])
        return diversity_score

    def compound_periodic_decay_weight(self, step, max_steps, p=2):
         """
         Calculate the compound periodic decay weight using predefined parameters.
         """
         poly_decay = (1 - step / max_steps) ** p

         return poly_decay 
    
    def calculate_combined_reward(self, pred_seq, target_seq, step, max_steps):
        # 计算相似性得分时需要处理填充字符
        similarity_score = self.sequence_similarity(pred_seq, target_seq)
        structure_pref_score = self.secondary_structure_preference(pred_seq)
        diversity_score = self.sequence_diversity(pred_seq)

        # 动态调整奖励的权重，随训练进程逐渐减小奖励
        time_weight = self.compound_periodic_decay_weight(step, max_steps, 2)
        
        grad_similarity = self.calculate_gradient_similarity(pred_seq, target_seq)
        grad_structure = self.calculate_gradient_structure(pred_seq, target_seq)
        grad_diversity = self.calculate_gradient_diversity(pred_seq, target_seq)
        
        # 更新权重
        self.weights['similarity'] = max(0, self.weights['similarity'] + 0.01 * grad_similarity)
        self.weights['structure'] = max(0, self.weights['structure'] + 0.01 * grad_structure)
        self.weights['diversity'] = max(0, self.weights['diversity'] + 0.01 * grad_diversity)

        
        # 计算总奖励
        combined_score = (self.weights['similarity'] * similarity_score +
                          self.weights['structure'] * structure_pref_score +
                          self.weights['diversity'] * diversity_score) * time_weight
        return combined_score
    
    def calculate_gradient_similarity(self, pred_seq, target_seq):
        epsilon = 1e-5  # 小的增量
        original_score = self.sequence_similarity(pred_seq, target_seq)  # 计算相似性得分
    
        # 改变序列的某个位置的氨基酸（扰动）
        perturbed_pred_seq = list(pred_seq)  # 将字符串转换为列表
        aa_to_replace = random.choice(list(perturbed_pred_seq))  # 随机选择一个氨基酸
        new_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')  # 从标准氨基酸中选择一个新的
        perturbed_pred_seq[perturbed_pred_seq.index(aa_to_replace)] = new_aa  # 替换氨基酸
    
        perturbed_score = self.sequence_similarity(''.join(perturbed_pred_seq), target_seq)
    
        # 数值梯度
        gradient = (perturbed_score - original_score) / epsilon
        return gradient
    
    def calculate_gradient_structure(self, pred_seq, target_seq):
        epsilon = 1e-5  # 小的增量
        original_score = self.secondary_structure_preference(pred_seq)  # 计算结构偏好得分

        # 通过扰动来改变氨基酸（这里选择一个位置的氨基酸）
        perturbed_pred_seq = list(pred_seq)  # 将字符串转换为列表
        aa_to_replace = random.choice(perturbed_pred_seq)  # 随机选择一个氨基酸
        new_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')  # 从标准氨基酸中选择一个新的
        perturbed_pred_seq[perturbed_pred_seq.index(aa_to_replace)] = new_aa  # 替换氨基酸

        perturbed_score = self.secondary_structure_preference(''.join(perturbed_pred_seq))

        # 数值梯度
        gradient = (perturbed_score - original_score) / epsilon
        return gradient
      
    def calculate_gradient_diversity(self, pred_seq, target_seq):
        epsilon = 1e-5  # 小的增量
        original_score = self.sequence_diversity(pred_seq)  # 计算多样性得分

        # 通过扰动来改变氨基酸（这里选择一个位置的氨基酸）
        perturbed_pred_seq = list(pred_seq)  # 将字符串转换为列表
        aa_to_replace = random.choice(perturbed_pred_seq)  # 随机选择一个氨基酸
        new_aa = random.choice('ACDEFGHIKLMNPQRSTVWY')  # 从标准氨基酸中选择一个新的
        perturbed_pred_seq[perturbed_pred_seq.index(aa_to_replace)] = new_aa  # 替换氨基酸

        perturbed_score = self.sequence_diversity(''.join(perturbed_pred_seq))

        # 数值梯度
        gradient = (perturbed_score - original_score) / epsilon
        return gradient

    #执行突变任务时请执行这个奖励函数
    def mutate_sequence_with_blosum62(self, wt_tokens, mutation_rate=0.1):
        wt_tokens = wt_tokens.clone()
        B, L = wt_tokens.size()
        var_tokens = wt_tokens.clone()
        for b in range(B):
            wt_seq_tokens = wt_tokens[b]
            num_mutations = max(1, int(L * mutation_rate))
            mutation_positions = np.random.choice(L, num_mutations, replace=False)

            for pos in mutation_positions:
                wt_aa = self.alphabet.get_tok(wt_seq_tokens[pos].item())
                # 忽略特殊标记
                if wt_aa in ['<pad>', '<cls>', '<eos>', '<mask>']:
                    continue

                # 根据BLOSUM62寻找最高分替换
                candidates = list(self.blosum62[wt_aa].items())
                # 移除自身残基
                candidates = [(aa, score) for aa, score in candidates if aa != wt_aa]
                # 按得分降序排序
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_aa = candidates[0][0]
                var_tokens[b, pos] = torch.tensor(self.alphabet.get_idx(best_aa), device=wt_tokens.device)
        return var_tokens
    
    def calculate_variant_effect(self, wt_tokens, var_tokens, batch):
        # 计算wt序列log概率
        wt_batch = copy.deepcopy(batch)
        wt_batch['prev_tokens'], wt_batch['prev_token_mask'] = self.inject_noise(
            wt_tokens, wt_batch['coord_mask'], noise=self.hparams.generator.noise)
        wt_logits = self.model(wt_batch)
        wt_probs = F.softmax(wt_logits, dim=-1)
        wt_seq_probs = wt_probs[torch.arange(wt_tokens.size(0)).unsqueeze(-1),
                                torch.arange(wt_tokens.size(1)).unsqueeze(0),
                                wt_tokens]
        wt_seq_probs = torch.clamp(wt_seq_probs, min=1e-9)
        wt_log_probs = torch.log(wt_seq_probs)  # [B, L]

        # 计算var序列log概率
        var_batch = copy.deepcopy(batch)
        var_batch['prev_tokens'], var_batch['prev_token_mask'] = self.inject_noise(
            var_tokens, var_batch['coord_mask'], noise=self.hparams.generator.noise)
        var_logits = self.model(var_batch)
        var_probs = F.softmax(var_logits, dim=-1)
        var_seq_probs = var_probs[torch.arange(var_tokens.size(0)).unsqueeze(-1),
                                  torch.arange(var_tokens.size(1)).unsqueeze(0),
                                  var_tokens]
        var_seq_probs = torch.clamp(var_seq_probs, min=1e-9)
        var_log_probs = torch.log(var_seq_probs)  # [B, L]

        # 计算log概率差
        log_prob_diff = var_log_probs - wt_log_probs  # [B, L]

        # 仅对突变位置求和或平均
        mutated_positions = (var_tokens != wt_tokens).float()
        variant_effect_scores = (log_prob_diff * mutated_positions).sum(dim=-1) / (mutated_positions.sum(dim=-1) + 1e-9)
        return variant_effect_scores
    
    def calculate_sp_reward(self, var_tokens, batch):
        """
        计算序列的SP值。这里使用var_tokens作为输入序列进行打分。
        需要在tmp_batch中放入var_tokens为'tokens'，以模拟模型对该变异序列的概率分布计算。
        """
        tmp_batch = copy.deepcopy(batch)
        tmp_batch['tokens'] = var_tokens.clone()  # 用var_tokens作为目标序列
        tmp_batch['prev_tokens'], tmp_batch['prev_token_mask'] = self.inject_noise(
            var_tokens, tmp_batch['coord_mask'], noise=self.hparams.generator.noise
        )
        
        with torch.no_grad():
            logits = self.model(tmp_batch)  # [B, L, V]
            probs = F.softmax(logits, dim=-1)

        B, L = var_tokens.size()
        pred_probs = probs[torch.arange(B).unsqueeze(-1),
                           torch.arange(L).unsqueeze(0),
                           var_tokens]
        pred_probs = torch.clamp(pred_probs, min=1e-9)
        log_probs = torch.log(pred_probs)
        sp_scores = torch.mean(log_probs, dim=-1)  # [B]
        return sp_scores
 
    

    
    def should_update_policy(self, batch_idx):
        # 每100个batches更新一次
        return batch_idx % 100 == 0
    
    # -------# Training #-------- #
    @torch.no_grad()
    def inject_noise(self, tokens, coord_mask, noise=None, sel_mask=None, mask_by_unk=False):
        padding_idx = self.alphabet.padding_idx
        if mask_by_unk:
            mask_idx = self.alphabet.unk_idx
        else:
            mask_idx = self.alphabet.mask_idx

        def _full_mask(target_tokens):
            target_mask = (
                target_tokens.ne(padding_idx)  # & mask
                & target_tokens.ne(self.alphabet.cls_idx)
                & target_tokens.ne(self.alphabet.eos_idx)
            )
            # masked_target_tokens = target_tokens.masked_fill(~target_mask, mask_idx)
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & coord_mask
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        def _adaptive_mask(target_tokens):
            raise NotImplementedError

        noise = noise or self.hparams.noise

        if noise == 'full_mask':
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':
            masked_tokens = tokens
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        prev_tokens = masked_tokens
        prev_token_mask = prev_tokens.eq(mask_idx) & coord_mask
        # target_mask = prev_token_mask & coord_mask

        return prev_tokens, prev_token_mask  # , target_mask

    def step(self, batch):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        coords = batch['coords']
        coord_mask = batch['coord_mask']
        tokens = batch['tokens']

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, coord_mask, noise=self.hparams.learning.noise)
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = label_mask = prev_token_mask

        logits = self.model(batch)

        if isinstance(logits, tuple):
            logits, encoder_logits = logits
            # loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)
            # NOTE: use fullseq loss for pLM prediction
            loss, logging_output = self.criterion(
                logits, tokens,
                # hack to calculate ppl over coord_mask in test as same other methods
                label_mask=label_mask if self.stage == 'test' else None
            )
            encoder_loss, encoder_logging_output = self.criterion(encoder_logits, tokens, label_mask=label_mask)

            loss = loss + encoder_loss
            logging_output['encoder/nll_loss'] = encoder_logging_output['nll_loss']
            logging_output['encoder/ppl'] = encoder_logging_output['ppl']
        else:
            loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)

        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        
        # 计算每个周期的总批次数
        max_steps = self.trainer.num_training_batches
        batch_copy = copy.deepcopy(batch)
        if 'prev_tokens' not in batch:
            # 正确访问padding_idx
            padding_idx = self.reinforce.policy_net.policy_net.padding_idx
            batch['prev_tokens'] = torch.full((batch['tokens'].size(0), batch['tokens'].size(1)), padding_idx, dtype=torch.long, device=device)
        logits = self.model(batch)
        
        loss, logging_output = self.step(batch)
        
        pred_tokens = self.forward(batch, return_ids=True)  # 假设返回的是预测的token indices;使用相似性、二级结构偏好，序列多样性时使用
        decoded_sequences = self.alphabet.decode(pred_tokens)
        batch['decoded_sequences'] = decoded_sequences
        
       
    
        # 使用REINFORCE的策略网络来选择动作，并获取日志概率
        #action, log_prob = self.policy_net.sample_action(batch)
        action, log_prob = self.reinforce.take_action(batch)
        self.reinforce.saved_log_probs.append(log_prob)
        reward = self.calculate_combined_reward(decoded_sequences, batch['seqs'], batch_idx, max_steps)
        self.reinforce.store_reward(reward)
        
        
        # log train metrics
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/loss', loss, on_step=True, prog_bar=True, logger=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)
        
        # 检查是否需要更新策略网络（例如，每个epoch结束时）
        if self.should_update_policy(batch_idx):
            self.reinforce.update()
            self.reinforce.rewards.clear()
            self.reinforce.saved_log_probs.clear()
        return {"loss": loss}
    
    def should_update_policy(self, batch_idx):
        # 这里可以定义触发更新的条件，例如每个epoch或固定批次数
        return (batch_idx + 1) % 100 == 0  # 每100个batch更新一次
    
    # -------# Evaluating #-------- #
    def on_test_epoch_start(self) -> None:
        self.hparams.noise = 'full_mask'

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log other metrics
        sample_size = logging_output['sample_size']
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        if self.stage == 'fit':
            pred_outs = self.predict_step(batch, batch_idx)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        eval_ppl = torch.exp(eval_nll_loss)

        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{log_key}/ppl", eval_ppl, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)
            self.log("val/ppl_best", self.val_ppl_best.compute(), on_epoch=True, prog_bar=True)

            self.predict_epoch_end(results=None)

        super().validation_epoch_end(outputs)

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        # In testing, remove target tokens to ensure no data leakage!
        # or you can just use the following one if you really know what you are doing:
        #   tokens = batch['tokens']
        tokens = batch.pop('tokens')

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, batch['coord_mask'],
            noise=self.hparams.generator.noise,  # NOTE: 'full_mask' by default. Set to 'selected_mask' when doing inpainting.
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        output_tokens, output_scores = self.generator.generate(
            model=self.model, batch=batch,
            max_iter=self.hparams.generator.max_iter,
            strategy=self.hparams.generator.strategy,
            replace_visible_tokens=self.hparams.generator.replace_visible_tokens,
            temperature=self.hparams.generator.temperature
        )
        if not return_ids:
            return self.alphabet.decode(output_tokens)
        return output_tokens

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        coord_mask = batch['coord_mask']
        tokens = batch['tokens']
        
        pred_tokens = self.forward(batch, return_ids=True)
        #print("*******************************************************************************************************************************************************************")
        
        # 打印预测的蛋白质序列索引
        #print("Predicted token indices:", pred_tokens)
        
        # 如果需要看到实际的氨基酸序列，可以使用字母表来解码
        #decoded_sequences = self.alphabet.decode(pred_tokens)
        #print("Decoded Predicted Sequences:", decoded_sequences)
        # NOTE: use esm-1b to refine
        # pred_tokens = self.esm_refine(
        #     pred_ids=torch.where(coord_mask, pred_tokens, prev_tokens))
        # # decode(pred_tokens[0:1], self.alphabet)

        if log_metrics:
            # per-sample accuracy
            recovery_acc_per_sample = metrics.accuracy_per_sample(pred_tokens, tokens, mask=coord_mask)
            self.acc_median.update(recovery_acc_per_sample)

            # # global accuracy
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=coord_mask)
            self.acc.update(recovery_acc, weight=coord_mask.sum())

        results = {
            'pred_tokens': pred_tokens,
            'names': batch['names'],
            'native': batch['seqs'],
            'recovery': recovery_acc_per_sample,
            'sc_tmscores': np.zeros(pred_tokens.shape[0])
        }


        if self.hparams.generator.eval_sc:
            torch.cuda.empty_cache()
            sc_tmscores = self.eval_self_consistency(pred_tokens, batch['coords'], mask=tokens.ne(self.alphabet.padding_idx))
            results['sc_tmscores'] = sc_tmscores

        return results

    def predict_epoch_end(self, results: List[Any]) -> None:
        log_key = 'test' if self.stage == 'test' else 'val'

        acc = self.acc.compute() * 100
        self.acc.reset()
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        acc_median = torch.median(self.acc_median.compute()) * 100
        self.acc_median.reset()
        self.log(f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.acc_best.update(acc)
            self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)

            self.acc_median_best.update(acc_median)
            self.log(f"{log_key}/acc_median_best", self.acc_median_best.compute(), on_epoch=True, prog_bar=True)
        else:
            if self.hparams.generator.eval_sc:
                import itertools
                sc_tmscores = list(itertools.chain(*[result['sc_tmscores'] for result in results]))
                self.log(f"{log_key}/sc_tmscores", np.mean(sc_tmscores), on_epoch=True, prog_bar=True)
            self.save_prediction(results, saveto=f'./test_tau{self.hparams.generator.temperature}.fasta')

    def save_prediction(self, results, saveto=None):
        save_dict = {}
        if saveto:
            saveto = os.path.abspath(saveto)
            log.info(f"Saving predictions to {saveto}...")
            fp = open(saveto, 'w')
            fp_native = open('./native.fasta', 'w')

        for entry in results:
            for name, prediction, native, recovery, scTM in zip(
                entry['names'],
                self.alphabet.decode(entry['pred_tokens'], remove_special=True),
                entry['native'],
                entry['recovery'],
                entry['sc_tmscores'],
            ):
                save_dict[name] = {
                    'prediction': prediction,
                    'native': native,
                    'recovery': recovery
                }
                if saveto:
                    fp.write(f">name={name} | L={len(prediction)} | AAR={recovery:.2f} | scTM={scTM:.2f}\n")
                    fp.write(f"{prediction}\n\n")
                    fp_native.write(f">name={name}\n{native}\n\n")

        if saveto:
            fp.close()
            fp_native.close()
        return save_dict

    def esm_refine(self, pred_ids, only_mask=False):
        """Use ESM-1b to refine model predicted"""
        if not hasattr(self, 'esm'):
            import esm
            self.esm, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            # self.esm, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_batcher = self.esm_alphabet.get_batch_converter()
            self.esm.to(self.device)
            self.esm.eval()

        mask = pred_ids.eq(self.alphabet.mask_idx)

        # _, _, input_ids = self.esm_batcher(
        #     [('_', seq) for seq in decode(pred_ids, self.alphabet)]
        # )
        # decode(pred_ids, self.alphabet)
        # input_ids = convert_by_alphabets(pred_ids, self.alphabet, self.esm_alphabet)

        input_ids = pred_ids
        results = self.esm(
            input_ids.to(self.device), repr_layers=[33], return_contacts=False
        )
        logits = results['logits']
        # refined_ids = logits.argmax(-1)[..., 1:-1]
        refined_ids = logits.argmax(-1)
        refined_ids = convert_by_alphabets(refined_ids, self.esm_alphabet, self.alphabet)

        if only_mask:
            refined_ids = torch.where(mask, refined_ids, pred_ids)
        return refined_ids

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def eval_self_consistency(self, pred_ids, positions, mask=None):
        pred_seqs = self.alphabet.decode(pred_ids, remove_special=True)

        # run_folding:
        sc_tmscores = []
        with torch.no_grad():
            output = self._folding_model.infer(sequences=pred_seqs, num_recycles=4)
            pred_seqs = self.alphabet.decode(output['aatype'], remove_special=True)
            for i in range(positions.shape[0]):
                pred_seq = pred_seqs[i]
                seqlen = len(pred_seq)
                _, sc_tmscore = metrics.calc_tm_score(
                    positions[i, 1:seqlen + 1, :3, :].cpu().numpy(),
                    output['positions'][-1, i, :seqlen, :3, :].cpu().numpy(),
                    pred_seq, pred_seq
                )
                sc_tmscores.append(sc_tmscore)
        return sc_tmscores


def convert_by_alphabets(ids, alphabet1, alphabet2, relpace_unk_to_mask=True):
    sizes = ids.size()
    mapped_flat = ids.new_tensor(
        [alphabet2.get_idx(alphabet1.get_tok(ind)) for ind in ids.flatten().tolist()]
    )
    if relpace_unk_to_mask:
        mapped_flat[mapped_flat.eq(alphabet2.unk_idx)] = alphabet2.mask_idx
    return mapped_flat.reshape(*sizes)
