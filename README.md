# IncSAR: A dual fusion incremental learning framework for SAR Target Recognition
The implementation of IEEE Access [paper](https://ieeexplore.ieee.org/document/10838563) **IncSAR: A dual fusion incremental learning framework for SAR Target Recognition** (You can also read our paper [here](https://mever.gr/publications/IncSAR_A_Dual_Fusion_Incremental_Learning_Framework_for_SAR_Target_Recognition.pdf).)

If you use any code of this repo, please consider citing our work:
```
@ARTICLE{Karantaidis2025,
  author={Karantaidis, G. and Pantsios, A. and Kompatsiaris, I. and Papadopoulos, S.},
  journal={IEEE Access}, 
  title={IncSAR: A Dual Fusion Incremental Learning Framework for SAR Target Recognition}, 
  year={2025},
  volume={13},
  pages={12358-12372},
  doi={10.1109/ACCESS.2025.3528633}}

```
## Pipeline

<img src='images/diagram.png' width='680' height='395'>

## Install
```
conda create -n incsar python=3.9
conda activate incsar
pip install -r requirements.txt
```
## Dependencies 
This code is implemented in PyTorch, and we perform the experiments under the following environment settings:
```
torch==2.0.1
torchvision==0.15.2
timm==0.6.12
Pillow==10.3.0
scikit_learn
scipy
tqdm
numpy==1.26.4
```

## Results
<img src='images/results.png'>

## Dataset preparation
- Create a folder "datasets/" under the root directory.
- MSTAR: download the folder MSTAR from [link](https://itigr-my.sharepoint.com/:f:/g/personal/karantai_iti_gr/EkURhx1iLLZPoLmgp02-v4IBq6AXDQrOw7064ZtjvlrW4A?e=4dt8hb) and place them into the 'datasets/' folder.
- MSTAR_OPENSAR: download the folder MSTAR_OPENSAR from [link](https://itigr-my.sharepoint.com/:f:/g/personal/karantai_iti_gr/EkURhx1iLLZPoLmgp02-v4IBq6AXDQrOw7064ZtjvlrW4A?e=4dt8hb) and place them into the 'datasets/' folder.
- SAR-AIRcraft-1.0: download the folder AIRCRAFT [link](https://itigr-my.sharepoint.com/:f:/g/personal/karantai_iti_gr/EkURhx1iLLZPoLmgp02-v4IBq6AXDQrOw7064ZtjvlrW4A?e=4dt8hb) and place them into the 'datasets/' folder.

## Run experiments: 
### Basic experiments
```
bash ./exps/basic_experiments/basic_experiments.sh
```
### Cross domain experiments
```
bash ./exps/cross_domain/cross_domain.sh
```
### Ablation Studies
- Contribution analysis of IncSAR module
```
bash ./exps/exps_ablation_components/ablation_components.sh
```
- Comparative analysis of backbone network
```
bash ./exps/exps_ablation_backbones/ablation_backbones.sh
```
- IncSAR evaluation on limited data scenario
```
bash ./exps/exps_ablation_portion/ablation_portion.sh
```

## Acknowledgments 
We thank the following repos providing helpful components/functions in our work.
- [PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- [RanPAC](https://github.com/RanPAC/RanPAC/)
- [TinyViT](https://github.com/wkcn/TinyViT)
- [SSF](https://github.com/dongzelian/SSF)

```

## Contact
If there are any questions, please feel free to contact the author George Karantaidis (karantai@iti.gr).
