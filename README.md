# Object DGCNN & DETR3D

This repo contains the implementations of Object DGCNN (https://arxiv.org/abs/2110.06923) and DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.  

### Prerequisite

1. mmcv (https://github.com/open-mmlab/mmcv)

2. mmdet (https://github.com/open-mmlab/mmdetection)

3. mmseg (https://github.com/open-mmlab/mmsegmentation)

4. mmdet3d (https://github.com/open-mmlab/mmdetection3d)

### Data
1. Follow the mmdet3d to process the data.

### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/ 

2. For example, to train Object-DGCNN with pillar on 8 GPUs, please use

`tools/dist_train.sh projects/configs/obj_dgcnn/pillar.py 8`

### Evaluation using pretrained models
1. Download the weights accordingly.  

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[DETR3D, ResNet101 w/ DCN](./projects/configs/detr3d/detr3d_res101_gridmask.py)|34.7|42.2|[model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing)|
|[above, + CBGS](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)|34.9|43.4|[model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing)|
|[DETR3D, VoVNet on trainval, evaluation on test set](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py)| 41.2 | 47.9 |[model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing)|

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[Object DGCNN, pillar](./projects/configs/obj_dgcnn/pillar.py)|53.2|62.8|[model](https://drive.google.com/file/d/1nd6-PPgdb2b2Bi3W8XPsXPIo2aXn5SO8/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1A98dWp7SBOdMpo1fHtirwfARvpE38KOn/view?usp=sharing)|
|[Object DGCNN, voxel](./projects/configs/obj_dgcnn/voxel.py)|58.6|66.0|[model](https://drive.google.com/file/d/1zwUue39W0cAP6lrPxC1Dbq_gqWoSiJUX/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1pjRMW2ffYdtL_vOYGFcyg4xJImbT7M2p/view?usp=sharing)|


2. To test, use  
`tools/dist_test.sh projects/configs/obj_dgcnn/pillar_cosine.py /path/to/ckpt 8 --eval=bbox`

 
If you find this repo useful for your research, please consider citing the papers

```
@inproceedings{
   obj-dgcnn,
   title={Object DGCNN: 3D Object Detection using Dynamic Graphs},
   author={Wang, Yue and Solomon, Justin M.},
   booktitle={2021 Conference on Neural Information Processing Systems ({NeurIPS})},
   year={2021}
}
```

```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```
### extra
①. 利用Resnet101 + fpn提取6张环视图像特征，获得1/4, 1/8, 1/16, 1/32, 4个不同尺度的输出(注意这里6张图的输入方式采用将Batch 和 N（camera_nums）拼接在一起的方式)
②. 预设900个object_querys(类似于2D中的priobox先验框)， 拆分object query为query和query_pos, 利用全连接将query_pos的维度由[900, 256]映射到[900, 3], 此时就获得了BEV空间3D reference point (x， y， z)的参考点。
③. 进入transformer decoder，共有6层decoder layer，其中在每层layer之中，令q=k=v=query，即所有的object query之间先做self-attention来相互交互获取全局信息并避免多个query收敛到同个物体。
④. 将预测的3D reference point左乘转换矩阵， 除以深度Zc，转换到二维的图像坐标系， 获得2D reference point。
⑤. 预测的3D reference point投影回2D中，可能无对应的点或者在当前相机下不可见，因此使用一个mask 表示3D reference point是否在当前相机位中。
⑥. 遍历fpn输出的四个特征层，利用2D reference point中的位置信息，在特征层中进行grid_sample（双线性插值）采样，获得与2D reference point对应的图像特征。
⑦. query作为attention权重，与图像特征进行cross-attention。
⑧. 用取到的特征去 refine（优化） 3D reference point，refine 的方式也非常简单粗暴，直接相加即可。
⑨. 利用全连接输出回归预测分支与分类预测分支
⑩. 匈牙利算法进行二分图匹配，获得正负样本，计算分类损失（focal loss）、回归损失（L1 loss）
### 采样投影机制
DETR3D首先根据object query预测N NN个参考点，然后利用相机参数将参考点反投影回图像，对2D图像特征进行采样，最后根据采样得到的2D图像特征预测3D目标信息
https://zhuanlan.zhihu.com/p/430198800
https://blog.csdn.net/qq_42250207/article/details/131833480?ops_request_misc=&request_id=&biz_id=102&utm_term=detr3d&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-131833480.142^v99^pc_search_result_base7&spm=1018.2226.3001.4187
