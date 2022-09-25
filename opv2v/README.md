# <div align="center">**CoBEVT OPV2V Track**</div>
This repository contains the source code and data for our CoBEVT OPV2V track. The whole pipeline is based on [OpenCOOD(ICRA2022)](https://github.com/DerrickXuNu/OpenCOOD)

## <div align="center">**Data Preparation**</div>

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
pip install -r requirements.txt
python opencood/utils/setup.py build_ext --inplace
python setup.py develop
```