from dataclasses import dataclass

import torch
import torch.nn as nn
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import new_arange, sample_from_categorical
from byprot.datamodules.datasets.data_utils import Alphabet

from .decoder import MPNNSequenceDecoder
from .encoder import MPNNEncoder


@dataclass
class ProteinMPNNConfig:
    d_model: int = 128
    d_node_feats: int = 128
    d_edge_feats: int = 128
    k_neighbors: int = 48
    augment_eps: float = 0.0
    n_enc_layers: int = 3
    dropout: float = 0.1

    # decoder-only
    n_vocab: int = 22
    n_dec_layers: int = 3
    random_decoding_order: bool = True
    nar: bool = True
    crf: bool = False
    use_esm_alphabet: bool = False
    
#融合层定义
class FusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FusionLayer, self).__init__()
        self.transform = nn.Sequential(                   #定义了一个顺序模型 nn.Sequential，它是 PyTorch 中用于包装多个层的容器，使得这些层可以按顺序执行
            nn.Linear(input_dim, output_dim),             #定义了一个线性层，将输入的特征从 input_dim 维转换到 output_dim 维。
            nn.ReLU(),                                    #一个激活层，使用了 ReLU (Rectified Linear Unit) 激活函数，用于增加模型的非线性能力，帮助解决梯度消失问题，并促进更快的训练。
            nn.Linear(output_dim, output_dim)             #再次将特征进行线性变换，输入和输出维度都是 output_dim，这可以进一步提炼特征。
        )

    def forward(self, x):
        return self.transform(x)                          #将输入数据 x 传递给之前定义的 self.transform 层序列，执行定义好的层序列操作，并返回最终的输出。

@register_model('protein_mpnn_cmlm')
class ProteinMPNNCMLM(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ProteinMPNNConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.encoder = MPNNEncoder(
            node_features=self.cfg.d_node_feats,
            edge_features=self.cfg.d_edge_feats,
            hidden_dim=self.cfg.d_model,
            num_encoder_layers=self.cfg.n_enc_layers,
            k_neighbors=self.cfg.k_neighbors,
            augment_eps=self.cfg.augment_eps,
            dropout=self.cfg.dropout
        )
        
        self.fusion_layer = FusionLayer(cfg.d_model, cfg.d_model)  # Adding fusion layer

        if self.cfg.use_esm_alphabet:
            alphabet = Alphabet('esm', 'cath')
            self.padding_idx = alphabet.padding_idx
            self.mask_idx = alphabet.mask_idx
        else:
            alphabet = None
            self.padding_idx = 0
            self.mask_idx = 1

        self.decoder = MPNNSequenceDecoder(
            n_vocab=self.cfg.n_vocab,
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_dec_layers,
            random_decoding_order=self.cfg.random_decoding_order,
            dropout=self.cfg.dropout,
            nar=self.cfg.nar,
            crf=self.cfg.crf,
            alphabet=alphabet
        )

    def _forward(self, coords, coord_mask, prev_tokens, token_padding_mask=None, target_tokens=None, return_feats=False, **kwargs):
        coord_mask = coord_mask.float()
        encoder_out = self.encoder(X=coords, mask=coord_mask)

        logits, feats = self.decoder(
            prev_tokens=prev_tokens,
            memory=encoder_out, memory_mask=coord_mask,
            target_tokens=target_tokens,
            **kwargs
        )

        if return_feats:
            return logits, feats
        return logits

    def forward(self, batch, return_feats=False, **kwargs):
        coord_mask = batch['coord_mask'].float()

        encoder_out = self.encoder(
            X=batch['coords'],
            mask=coord_mask,
            residue_idx=batch.get('residue_idx', None),
            chain_idx=batch.get('chain_idx', None)
        )
        
         # Apply fusion layer after encoder output
        fused_output = self.fusion_layer(encoder_out['node_feats'])

        logits, feats = self.decoder(
            prev_tokens=batch['prev_tokens'],
            memory=encoder_out, 
            memory_mask=coord_mask,
            target_tokens=batch.get('tokens'),
            **kwargs
        )

        if return_feats:
            return logits, feats
        return logits

    def forward_encoder(self, batch):
        encoder_out = self.encoder(
            X=batch['coords'],
            mask=batch['coord_mask'].float(),
            residue_idx=batch.get('residue_idx', None),
            chain_idx=batch.get('chain_idx', None)
        )
        encoder_out['coord_mask'] = batch['coord_mask'].float()

        return encoder_out

    def forward_decoder(self, prev_decoder_out, encoder_out, need_attn_weights=False):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = output_tokens.eq(self.mask_idx)  # & coord_mask

        logits, _ = self.decoder(
            prev_tokens=output_tokens,
            memory=encoder_out,
            memory_mask=encoder_out['coord_mask'].float(),
        )
        # log_probs = torch.log_softmax(logits, dim=-1)
        # _scores, _tokens = log_probs.max(dim=-1)
        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            step=step + 1,
            max_step=max_step,
            history=history
        )

    def initialize_output_tokens(self, batch, encoder_out):
        # mask = encoder_out.get('coord_mask', None)

        prev_tokens = batch['prev_tokens']
        lengths = prev_tokens.ne(self.padding_idx).sum(1)

        initial_output_tokens = torch.full_like(prev_tokens, self.padding_idx)
        initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.mask_idx)

        # if mask is not None:
        #     initial_output_tokens = torch.where(
        #         ~mask, prev_tokens, initial_output_tokens
        #     )
        # initial_output_tokens = prev_tokens.clone()

        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores