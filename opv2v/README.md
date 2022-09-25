# <div align="center">**CoBEVT OPV2V Track**</div>
This repository contains the source code and data for our CoBEVT OPV2V track. The whole pipeline is based on [OpenCOOD(ICRA2022)](https://github.com/DerrickXuNu/OpenCOOD)

## <div align="center">**Data Preparation**</div>
1. Download OPV2V origin data and structure it as required. See [OpenCOOD data tutorial](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html) for more detailed insructions.
2. After organize the data folders, download the `additional.zip` from [this url](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu?usp=sharing). This file contains BEV semantic segmentation labels that origin OPV2V data does not include.
3. The `additional` folder has the same structure of original OPV2V dataset. So unzip `additional.zip` and merge them with original opv2v data.
4. Remove scenario `opv2v/train/2021_09_09_13_20_58`, as this scenario has some bug for camera data.
## <div align="center">**Installation**</div>

```bash
# Clone repo
git clone https://github.com/DerrickXuNu/CoBEVT.git

cd CoBEVT/opv2v

# Setup conda environment
conda create -y --name cobevt python=3.7

conda activate cobevt
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install dependencies

python opencood/utils/setup.py build_ext --inplace
python setup.py develop
```

## <div align="center">**Visialization**</div>