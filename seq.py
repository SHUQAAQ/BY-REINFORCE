from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer
# 1. instantialize designer
exp_path = "/root/autodl-tmp/ByProt/logs/fixedbb/cath_4.2/protein_mpnn_cmlm"
cfg = Cfg(
    cuda=True,
    generator=Cfg(
        max_iter=1,
        strategy='mask_predict',
        temperature=0,
        eval_sc=False,  
    )
)
designer = Designer(experiment_path=exp_path, cfg=cfg)


pdb_path = "/root/autodl-tmp/ByProt/pdb/5ggs.pdb"
designer.set_structure(pdb_path)

# 提取原始序列
native_seq = designer._structure['seq']
print("原始序列:", native_seq)
