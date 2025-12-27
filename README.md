# UA-HLR
This repository contains the official implementation of "UA-HLR: Uncertainty-Aware Hierarchical Learning for Robust Source-Free Domain Adaptation".
---
##  Requirements
- Python >= 3.7  
- PyTorch >= 1.13  
- torchvision >= 0.14  
- numpy
- scipy
- scikit-learn
- tqdm
- PyYAML
- tensorboard
- matplotlib
---
## Datasets
This repository supports the following datasets:
- Office-31  
- Office-Home  
- VISDA-C  
- DomainNet126  

You need to download the datasets and update the image paths in the corresponding `.txt` files under the `./data/` folder.  
In addition, the class name files for each dataset should also be placed under `./data/`.

---

## Getting started

### Source
The pre-trained weights of source models are available at this [link](https://drive.google.com/drive/folders/1-2ROsqftoto_5XHmj_Ubu7POV-V0Q69x).

### Target
After obtaining the source models, modify the `CKPT_DIR` in the `conf.py` to your source model directory. For office-31, office-home ,VISDA-C and domainnet126 simply run the following Python file with the corresponding config file to execute source-free domain adaptation.
```bash
python image_target_of_oh_vs.py --cfg "cfgs/office31/uahlr.yaml" SETTING.S 0 SETTING.T 1

python image_target_of_oh_vs.py --cfg "cfgs/office-home/uahlr.yaml" SETTING.S 0 SETTING.T 1

python image_target_of_oh_vs.py --cfg "cfgs/visda/uahlr.yaml" SETTING.S 0 SETTING.T 1

python image_target_of_oh_vs.py --cfg "cfgs/domainnet126/uahlr.yaml" SETTING.S 0 SETTING.T 1
```
We also provide the trained target model weights with proposed UA-HLR method reported in the paper, which can be downloaded from [here](https://drive.google.com/file/d/1pLikCw6jOit12nsyV81OZ03Vng3KNDtu/view).

## Acknowledgements
This work is built upon the [DIFO codebase](https://github.com/tntek/source-free-domain-adaptation).
We thank the authors of DIFO for their open-source implementation.