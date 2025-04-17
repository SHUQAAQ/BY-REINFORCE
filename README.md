
# 🧬 Protein Design with ProteinMPNN (CMLM)

## 📦 1. Install Environment

```bash
git clone --recurse-submodules https://github.com/SHUQAAQ/BY-REINFORCE.git

conda create -n ${env_name} python=3.7 
conda activate ${env_name}
```

## 🔧 2. Install Dependencies

```bash
bash install.sh

pip install lmdb
pip install tmtools
pip install python-Levenshtein

pip install -e .
pip install -e ./vendor/esm
```

> You may also manage dependencies with a `requirements.txt` file.

---

## 📁 3. Data Preparation

### Download Preprocessed CATH Datasets

- **CATH 4.2** dataset provided by:  
  [Generative Models for Graph-Based Protein Design (NeurIPS 2019)](https://papers.nips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)

- **CATH 4.3** dataset provided by:  
  [Learning inverse folding from millions of predicted structures (bioRxiv)](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1)

Go check configs/datamodule/cath_4.*.yaml and set data_dir to the path of the downloaded CATH data.
Download with:

```bash
bash scripts/download_cath.sh
```

---

## 🏋️‍♂️ 4. Training

Train NAR ProteinMPNN using Conditional Masked Language Modeling (CMLM):

```bash
export CUDA_VISIBLE_DEVICES=0
# or use multi-gpu training when you want:
# export CUDA_VISIBLE_DEVICES=0,1

exp=fixedbb/protein_mpnn_cmlm  
dataset=cath_4.2
name=fixedbb/${dataset}/protein_mpnn_cmlm

python ./train.py \
    experiment=${exp} datamodule=${dataset} name=${name} \
    logger=tensorboard trainer=ddp_fp16
```

Or use the full explicit command (equivalent to the above):

```bash
python train.py experiment=fixedbb/protein_mpnn_cmlm \
    datamodule=cath_4.2 \
    name=fixedbb/cath_4.2/protein_mpnn_cmlm \
    logger=tensorboard \
    trainer=ddp_fp16
```


---

**Hardware Recommendations**:

This project is recommended to run on **NVIDIA RTX 4090 or higher-end GPUs** (e.g., **A100**, **H100**).

A **single GPU with at least 24GB of memory** is required. It is recommended to use **AMP mixed-precision training** (`ddp_fp16`) to reduce memory consumption.

For multi-GPU training, please properly set `CUDA_VISIBLE_DEVICES`.

---

### Optional Training Flags

| Argument              | Description                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `experiment`          | Experiment config file under `configs/experiment/`                                               |
| `datamodule`          | Dataset config file under `configs/datamodule/`                                                  |
| `name`                | Output name and folder path (e.g., `run/logs/${name}`)                                           |
| `logger`              | Logger to use (e.g., `tensorboard`)                                                              |
| `train.force_restart` | Set to `true` to force re-training even if checkpoints exist                                     |

---

## 🧪 5. Example Inference:Designing sequences from a pdb file using a trained model in Notebook
### Example 1:seq-design
```python
from byprot.utils.config import compose_config as Cfg
from byprot.tasks.fixedbb.designer import Designer

# 1. instantialize designer
exp_path = "/root/research/projects/BY-REINFORCE/run/logs/fixedbb/cath_4.2/protein_mpnn_cmlm"
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
pdb_path = "/root/research/projects/BY-REINFORCE/testpdb/5ggs.pdb"
designer.set_structure(pdb_path)

# 3. generate sequence from the given structure
designer.generate()

# 4. calculate evaluation metircs
designer.calculate_metrics()
## prediction: ETIEQSGSKYKKPGDSVTLSCKGSKEDFENHYFYWVRQKPGEGLTFLGGINPLNGGTNYNERYKSRVTLSVDKSTHTSYMTISNLKEDDTAVYYCAIVDKRFFTGFFLFGKGTRVIVSSNEVTGPRVYPLRPSEQSTNGGTAALGCLVDDYFPEPVTVTWNDGALTEGVHTFPAVLTASGLYTQTSTVTIPASSLGTKTYVCNVTHEPSGSHQSLEVNPPIVLLQYPPVLSKKPGEKAVLHCVASESVSKDGLTYMSWYKQKPGEAPELLIFNSSFRAKGVPERFSGSGSGTDFSLTIESLQPEDFATYYCMHSADLPITFSGGTKVEVKTETTAPSVYIFPPSEEQLKEGVAVVTCFASDFYPKNIKLEWYVDNELVSGRSLSSTTEQDAVDHTYSLSTLLRLSTEEYESHTVFACNVTHSGLDEPLTKSFNRGESISLTQSGAKFKKPGESVTVSCKASGFNFTDHYVSWVRQKPGEGLTYLGGINPSDGGTNYNEEYKSRVTLTYDKANNTSYLTLSNLKPDDTATYYCAITDKRWFTGLTHWGQGTRVIVSSAEVTGPRVYPLSPSEQSTSGGEAVLGCLVSDYFPEPVTVTWNDGALTEGVHTFPAVLTDSGLYTQTSVITVPSSSLGKETYVCNVTHEPSNTHQSLLVTPGIVLLESPSVLSLKPGEKATLKCVASESVSEDGLTYLSWFKQKPGKAPELLIFQASFRAPGIPSRYSGSGSGTDFTLTITSLKPEDFATYYCMHSYTLPITFGGGTLVVRKTETKAPTVFIFPPSEEQLKRGVARVVCLLLDFYPRDVKVTWYIDNELVSGNSQSSTTEQDSTDHTYSLSSILTLSTEEYEKCRVFACVVSHSGLSEPVTKSFVVGKGPPTFSPKLLTVKEGDDGTFTCTFSHDKERYVLNFYRLSPDKYDLVLTSYPENVSQPGRDDRYVMEKLPDGDSYNCTIKNAQKDDSGTYSCGAVDLEPRDTTLYSDTATLEVTTGPSPPTFTPKLLVVSKGDNATFTCTYNGDSDKYVLNFYRMSPENEDYVLTSYPKNVSQPGYDERYVLTKLPDGQTYDATIKNAQENDAGTYSCGVVDLKPADKVLYSETVTIE
## recovery: 0.5972727272727273
```
### Example 2 :Inpainting ** For some use cases, you may want to do inpainting on some segments of interest only while the rest of the protein remains the same (e.g., designing antibody CDRs). Here is a simple example with inpaint interface:

```python
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

pdb_path = "/root/autodl-tmp/BY-REINFORCE/testpdb/1a2y.pdb"
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
#[['IELDQSPSSI'], ['TTLLADGVPGRISGSGEGTRFSLKIEDIRPQDFGKYWCVHFYDYPQTFGDG']]
#[['VTLDQSPLNI'], ['TTHLADGIPDRFSGNGSGKSFSLTIFQIQPKDFGYYWCMHFELTPRSFGGG']]
#[['TQLTVSPKTI'], ['TTHLAEGVPKRFTGKGAGTTYSLTIMEIQPEDFGSYFCRNFYDTPKSFGNG']]
#[['IALRQDPKSI'], ['VTRLAKGVPERFTGSGSGTQFSLTISNIQPEDFGNYWCMDFAAYPLHYGNG']]
#[['IQLRQTPASI'], ['TSNLATGIPERFTGSGSGKDFSLTIENIKPDDFGSYYCMNHFDYPRSFGSG']]
#Original Segments:
#[['IVLTQSPASL'], ['TTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGG']]
```
