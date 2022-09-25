# <div align="center">**CoBEVT nuScenes Track**</div>

This repository contains the source code and data for our CoBEVT nuScenes track. The whole pipeline is based on [CVT(CVPR2022)](https://github.com/bradyz/cross_view_transformers)


## <div align="center">**Installation**</div>

```bash
# Clone repo
git clone https://github.com/DerrickXuNu/CoBEVT.git

cd CoBEVT/nuScenes

# Setup conda environment
conda create -y --name sinbevt python=3.8

conda activate sinbevt
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## <div align="center">**Data**</div>


Documentation:
* [Dataset setup](docs/dataset_setup.md)
* [Label generation](docs/label_generation.md) (optional)

<br/>

Download the original datasets and our generated map-view labels

| | Dataset | Labels |
| :-- | :-- | :-- |
| nuScenes | [keyframes + map expansion](https://www.nuscenes.org/nuscenes#download) (60 GB) | [cvt_labels_nuscenes.tar.gz](https://www.cs.utexas.edu/~bzhou/cvt/cvt_labels_nuscenes.tar.gz) (361 MB) |

<br/>

The structure of the extracted data should look like the following

```
/datasets/
├─ nuscenes/
│  ├─ v1.0-trainval/
│  ├─ v1.0-mini/
│  ├─ samples/
│  ├─ sweeps/
│  └─ maps/
│     ├─ basemap/
│     └─ expansion/
└─ cvt_labels_nuscenes/
   ├─ scene-0001/
   ├─ scene-0001.json
   ├─ ...
   ├─ scene-1000/
   └─ scene-1000.json
```

When everything is setup correctly, check out the dataset with

```bash
python3 scripts/view_data.py \
  data=nuscenes \
  data.dataset_dir=/media/datasets/nuscenes \
  data.labels_dir=/media/datasets/cvt_labels_nuscenes \
  data.version=v1.0-mini \
  visualization=nuscenes_viz \
  +split=val
```

# <div align="center">**Training**</div>

<div align="center">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://wandb.ai/site">
<img src="https://raw.githubusercontent.com/wandb/client/master/.github/wb-logo-lightbg.png" width="25%">
</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://hydra.cc">
<img src="https://raw.githubusercontent.com/facebookresearch/hydra/master/website/static/img/Hydra-Readme-logo2.svg" width="15%">
</a>
</div>

<br>

An average job of 50k training iterations takes ~8 hours.  
We trained model both on four A5000 and two A100(80G), and we found a total batch size of 16 will get the best results.

To train a model,

```bash
python scripts/train.py \
  +experiment=cvt_pyramid_axial_nuscenes_vehicle
  data.dataset_dir=/media/datasets/nuscenes \
  data.labels_dir=/media/datasets/cvt_labels_nuscenes
```

For more information, see

* `config/config.yaml` - base config
* `config/model/cvt_pyramid_axial.yaml` - model architecture (CVT + Pyramid FAX)
* `config/experiment/cvt_pyramid_axial_nuscenes_vehicle.yaml` - additional overrides

# <div align="center">**Benchmarking**</div>
To benchmark the inference speed, run the following command:
```bash
python scripts/benchmark.py \
  +experiment=cvt_pyramid_axial_nuscenes_vehicle
  data.dataset_dir=/media/datasets/nuscenes \
  data.labels_dir=/media/datasets/cvt_labels_nuscenes
```



## <div align="center">**Additional Information**</div>

### **Acknoledgement**
We would like to sinsere thank [CVT(CVPR2022)](https://github.com/bradyz/cross_view_transformers) for their awesome training pipeline, which makes our implementation much easier.
c