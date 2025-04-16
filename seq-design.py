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

# 2. load structure from pdb file
pdb_path = "/root/autodl-tmp/ByProt/pdb/5ggs.pdb"
designer.set_structure(pdb_path)

# 3. generate sequence from the given structure
designer.generate()

# 4. calculate evaluation metircs
designer.calculate_metrics()
## prediction: SSYNPPILLLGPFAEELEEELVEENPERAGRPVPFTTEPPSPDETEGETYLYISSLEEAEELIESNRFLEAGEENNELVGISLEAIRSVARAGKLAILDTGGEAVEKLEEANIEPIVIFLVPKSVEDVRRVFPDLTEEEAEELTSEDEELLEEFKELLDAVVSGSTLEEVLEEIREVIEEASS
## recovery: 0.37158469945355194