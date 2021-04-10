# Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices

This repository contains the code (in PyTorch) for "[Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices](https://openaccess.thecvf.com/content/ACCV2020/papers/Chang_Attention-Aware_Feature_Aggregation_for_Real-time_Stereo_Matching_on_Edge_Devices_ACCV_2020_paper.pdf)" paper (ACCV 2020) by [Jia-Ren Chang](https://jiarenchang.github.io/), [Pei-Chun Chang](https://scholar.google.com/citations?user=eJUcMrQAAAAJ&hl=zh-TW) and [Yong-Sheng Chen](https://people.cs.nctu.edu.tw/~yschen/).

The codes mainly bring from [PSMNet](https://github.com/JiaRenChang/PSMNet/).

### Citation
```
@InProceedings{Chang_2020_ACCV,
    author    = {Chang, Jia-Ren and Chang, Pei-Chun and Chen, Yong-Sheng},
    title     = {Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```
### Train
As an example, use the following command to train a RTStereo on Scene Flow

```
python main.py --maxdisp 192 \
               --model RTStereoNet \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)
```

### Pretrained Model

KITTI 2015 Pretrained Model [Google Drive](https://drive.google.com/file/d/12EQKjntE_Vi6m9vpSzJRtuzDCRJRmYoV/view?usp=sharing)
