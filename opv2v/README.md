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
To quickly visualize a single sample of the data:
```shell
cd CoBEVT/opv2v
python opencood/visualization/visialize_camera.py [--scene ${SCENE_NUMBER} --sample ${SAMPLE_NUMBER}]
```
* `scene`: The ith scene in the data. Default: 4
* `sample`: The jth sample in the ith scene. Default: 10

## <div align="center">**Training**</div>
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint **on a single gpu**, run the following commonds:
```python
python opencood/tools/train_camera.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/opcamera/cobevt.yaml`.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
  
To train on **multiple gpus**, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train_camera.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}
```

## <div align="center">**Inference**</div>
To run pre-trained cobevt, please first download `cobevt` and `cobevt_static` pretrained weights from [this url](https://drive.google.com/drive/folders/1NLzyvMFxuv8Qy52q_OzcNsugTS5JacAT?usp=sharing) ,
and then put them under `opv2v/logs/`. 

Please run the following command for dynamic BEV map segmentation 
```python
python opencood/tools/inference_camera.py --model_dir opencood/logs/cobevt
```

Please run the following command for static BEV map segmentation (road+lane)
```python
python opencood/tools/inference_camera.py --model_dir opencood/logs/cobevt_static --model_type static
```

To merge the results from both static and dynamic models, please run the following command (please run the above two inference command first)
```python
python opencood/tools/merge_dynamic_static.py --dynamic_path opencood/logs/cobevt --static_path opencood/logs/cobevt_static --output_path merge_results
```

Note: When you want to run on test set, make sure change `validation_dir` in the yaml file to the testing folder.


