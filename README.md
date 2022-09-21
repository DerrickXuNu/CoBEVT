# CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers [CORL2022] 

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2207.02202.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)](https://arxiv.org/pdf/2207.02202.pdf)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)]()

This is the official implementation of CoRL2022 paper "CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers".
[Runsheng Xu](https://derrickxunu.github.io/), [Zhengzhong Tu](https://github.com/vztu), [Hao Xiang](https://xhwind.github.io/), [Wei Shao](https://www.linkedin.com/in/wei-shao-94972295/), [Bolei Zhou](https://boleizhou.github.io/), [Jiaqi Ma](https://mobility-lab.seas.ucla.edu/)

UCLA, UT-Austin

![teaser](images/CorpBEVT_Overview-1.png)

## Introduction
CoBEVT is the first generic multi-agent multi-camera perception framework that can cooperatively generate BEV
map predictions. The core component of CoBEVT, named fused axial
attention or FAX module,  can capture sparsely local and global spatial interactions across views and agents. We 
achieve SOTA performance both on [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) and [nuScenes](https://www.nuscenes.org/) dataset with **real-time performance**.

### Demo
<br>

<div align="center"><img src="images/nuscene.gif" width="75%"/></div>
<div align="center">
<b>nuScenes demo:</b>
Our CoBEVT can be used on **single-vehicle multi-camera** semantic BEV Segmentations.
</div>
<br>

<br>

<div align="center"><img src="images/opv2v.gif" width="75%"/></div>
<div align="center">
<b>OPV2V demo:</b>
Our CoBEVT can also be used for **multi-agent** BEV map prediction.
</div>
<br>

## Installation
The pipeline for nuScenes dataset and OPV2V dataset is different. Please refer to the specific folder for more details based on your research purpose.

:point_right: [nuScenes Users](nuScenes/README.md) <br/>
:point_right: [OPV2V Users](opv2v/README.md)

## Citation
 ```bibtex
@inproceedings{xu2022cobevt,
  author = {Runsheng Xu, Zhengzhong Tu, Hao Xiang, Wei Shao, Bolei Zhou, Jiaqi Ma},
  title = {CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers},
  booktitle={Conference on Robot Learning (CoRL)},
  year = {2022}}
```
