from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

exp_path = "/root/autodl-tmp/BY-REINFORCE/logs/fixedbb/cath_4.2/protein_mpnn_cmlm"
cfg = Cfg(
    cuda=True,
    generator=Cfg(
        max_iter=5,
        strategy='mask_predict', 
        temperature=0,
        eval_sc=False,  
    )
)
designer = Designer(experiment_path=exp_path,cfg=cfg)

pdb_path = "/root/autodl-tmp/BY-REINFORCE/pdb/1a2y.pdb"
designer.set_structure(pdb_path)

start_ids = [1,50]
end_ids = [10,100]

for i in range(5):
    out, ori_seg, designed_seg = designer.inpaint(
        start_ids=start_ids, end_ids=end_ids, 
        generator_args={'temperature': 1.0}
    )
    print(designed_seg)
print('Original Segments:')
print(ori_seg)