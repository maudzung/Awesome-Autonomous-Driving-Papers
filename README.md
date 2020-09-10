# Awesome-Autonomous-Driving-Papers
---
This repository provides awesome research papers for autonomous driving perception. <br>
I have tried my best to keep this repository up to date. If you do find a problem or have any suggestions, please raise this as 
an issue or make a pull request with information (format of the repo): Research paper title, datasets, metrics, objects, source code, publisher, and year.


This summary is categorized into:
- [Datasets](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#Datasets)
- [LiDAR-based 3D Object Detection](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#1-lidar-based-3d-object-detection)
    - [Single-stage detectors](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#11-single-stage-detectors)
    - [Two-stage detectors](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#12-two-stage-detectors)
- [Monocular Image-based 3D Object Detection](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#2-monocular-image-based-3d-object-detection)
- [LiDAR and RGB Images fusion](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#3-lidar-and-rgb-images-fusion)
- [Pseudo-LiDAR](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#4-pseudo-lidar)
- [Training tricks](https://github.com/maudzung/Awesome-Autonomous-Driving-Papers#5-training-tricks)

_**Abbreviations**_
- **AP-2D**: **A**verage **P**recision for 2D detection (on RGB-image space)
- **AP-3D**: **A**verage **P**recision for 3D detection
- **AP-BEV**: **A**verage **P**recision for Birds Eye View
- **AOS**: **A**verage **O**rientation **S**imilarity _(if 2D bounding box available)_

## Datasets
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [Waymo Open Dataset](https://waymo.com/open/)
- [nuScenes](https://www.nuscenes.org)
- [Lift](https://self-driving.lyft.com/level5/data/)

## 1. LiDAR-based 3D Object Detection

### 1.1 Single-stage detectors

<table>
  <tbody>
    <tr>
      <th width="25%">Research Paper</th>
      <th align="center" width="15%">Datasets</th>
      <th align="center" width="15%">Metrics</th>
      <th align="center" width="15%">Objects</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Publisher</th>
      <th align="center" width="10%">Year</th>
    </tr>  
    <tr>
      <td><a href='https://arxiv.org/pdf/2006.15505.pdf'> “HorizonLiDAR3D”: 1st Place Solution for Waymo Open Dataset Challenge - 3D Detection and Domain Adaptation </a></td>
      <td align="left"><ul><li> Waymo </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href=''> --- </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.pdf'> Structure Aware Single-stage 3D Object Detection from Point Cloud (SA-SSD) </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/skyhehe123/SA-SSD'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/2002.10187.pdf'> 3DSSD: Point-based 3D Single Stage Object Detector </a></td>
      <td align="left"><ul><li> KITTI </li><li> nuScenes </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/Jia-Research-Lab/3DSSD'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2019/papers/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.pdf'> LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving </a></td>
      <td align="left"><ul><li> KITTI </li><li> ATG4D </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">CVPR</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1812.05784.pdf'> PointPillars: Fast Encoders for Object Detection from Point Clouds </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/nutonomy/second.pytorch'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://www.mdpi.com/1424-8220/18/10/3337/htm'> SECOND: Sparsely Embedded Convolutional Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/traveller59/second.pytorch'> PyTorch </a></td>
      <td align="center">Sensors</td>
      <td align="center">2018</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1803.06199.pdf'> Complex-YOLO: an euler-region-proposal for real-time 3d object detection on point clouds </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/maudzung/Complex-YOLOv4-Pytorch'> PyTorch </a></td>
      <td align="center">ECCV</td>
      <td align="center">2018</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1808.02350v1.pdf'> YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> mAP-3D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/maudzung/YOLO3D-YOLOv4-PyTorch'> PyTorch </a></td>
      <td align="center">ECCV</td>
      <td align="center">2018</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf'> Pixor: Real-time 3d object detection from point clouds </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/ankita-kalra/PIXOR'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2018</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf'> VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/skyhehe123/VoxelNet-pytorch'> PyTorch </a><a href='https://github.com/tsinghua-rll/VoxelNet-tensorflow'> Tensorflow </a></td>
      <td align="center">CVPR</td>
      <td align="center">2018</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1611.08069.pdf'> 3D Fully Convolutional Network using PointCloud data for Vehicle Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AOS </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/yukitsuji/3D_CNN_tensorflow'> Tensorflow </a></td>
      <td align="center">IROS</td>
      <td align="center">2017</td>   
    </tr>
    
  </tbody>
</table>


### 1.2 Two-stage detectors

<table>
  <tbody>
    <tr>
      <th width="25%">Research Paper</th>
      <th align="center" width="15%">Datasets</th>
      <th align="center" width="15%">Metrics</th>
      <th align="center" width="15%">Objects</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Publisher</th>
      <th align="center" width="10%">Year</th>
    </tr>  
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.pdf'> PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li><li> Waymo </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/open-mmlab/OpenPCDet'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Point-GNN_Graph_Neural_Network_for_3D_Object_Detection_in_a_CVPR_2020_paper.pdf'> Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars (1 model), Pedestrians and Cyclists(1 model) </td>
      <td align="left"><a href='https://github.com/WeijingShi/Point-GNN'> Tensorflow </a></td>
      <td align="center">CVPR</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/2004.04962.pdf'> 3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href=''> --- </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1903.01864.pdf'> PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/sshaoshuai/PointRCNN'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1910.04093.pdf'> Patch Refinement - Localized 3D Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1908.11069.pdf'> StarNet: Targeted Computation for Object Detection in Point Clouds </a></td>
      <td align="left"><ul><li> KITTI </li><li> Waymo </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/ModelBunker/StarNet-PyTorch'> PyTorch </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1907.03670.pdf'> Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/open-mmlab/OpenPCDet'> PyTorch </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1908.02990.pdf'> Fast Point R-CNN </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">ICCV</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1907.10471.pdf'> STD: Sparse-to-Dense 3D Object Detector for Point Cloud </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">ICCV</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1901.08373.pdf'> Three-dimensional Backbone Network for 3D Object Detection in Traffic Scenes </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/Benzlxs/tDBN'> PyTorch </a></td>
      <td align="center">ICCV</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1903.01864.pdf'> Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li><li> SUN-RGBD </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/zhixinwang/frustum-convnet'> PyTorch </a></td>
      <td align="center">IROS</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf'> Frustum PointNets for 3D Object Detection from RGB-D Data </a></td>
      <td align="left"><ul><li> KITTI </li><li> SUN-RGBD </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/charlesq34/frustum-pointnets'> Tensorflow </a></td>
      <td align="center">CVPR</td>
      <td align="center">2018</td>   
    </tr>
    
  </tbody>
</table>

## 2. Monocular Image-based 3D Object Detection

<table>
  <tbody>
    <tr>
      <th width="25%">Research Paper</th>
      <th align="center" width="15%">Datasets</th>
      <th align="center" width="15%">Metrics</th>
      <th align="center" width="15%">Objects</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Publisher</th>
      <th align="center" width="10%">Year</th>
    </tr>  
    <tr>
      <td><a href='https://arxiv.org/pdf/2001.03343.pdf'> RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li><li> AOS </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/maudzung/RTM3D'> PyTorch </a></td>
      <td align="center">ECCV</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Stereo_R-CNN_Based_3D_Object_Detection_for_Autonomous_Driving_CVPR_2019_paper.pdf'> Stereo R-CNN based 3D Object Detection for Autonomous Driving </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D/BEV </li><li> AP-2D </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/srinu6/Stereo-3D-Object-Detection-for-Autonomous-Driving'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1907.06038.pdf'> M3D-RPN: Monocular 3D Region Proposal Network for Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D/BEV </li><li> AP-2D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/garrickbrazil/M3D-RPN'> PyTorch </a></td>
      <td align="center">ICCV</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1901.03446.pdf'> Mono3D++: Monocular 3D Vehicle Detection with Two-Scale 3D Hypotheses and Task Priors </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D/BEV </li><li> AP-2D </li></ul></td>
      <td align="left">Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2019</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1612.00496.pdf'> 3D Bounding Box Estimation Using Deep Learning and Geometry </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP </li></ul></td>
      <td align="left">Cars, Cyclists</td>
      <td align="left"><a href='https://github.com/skhadem/3D-BoundingBox'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2017</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_cvpr_2017/papers/Chabot_Deep_MANTA_A_CVPR_2017_paper.pdf'> Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li>  </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">CVPR</td>
      <td align="center">2017</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_cvpr_2017/papers/Chabot_Deep_MANTA_A_CVPR_2017_paper.pdf'> Deep MANTA: A Coarse-to-fine Many-Task Network for joint 2D and 3D vehicle analysis from monocular image </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-2D/3D </li><li> AOS </li></ul></td>
      <td align="left">Cars</td>
      <td align="left"><a href='https://github.com/krrish94/ICRA2017'> Link </a></td>
      <td align="center">ICRA</td>
      <td align="center">2017</td>   
    </tr>
  </tbody>
</table>

## 3. LiDAR and RGB Images Fusion

<table>
  <tbody>
    <tr>
      <th width="25%">Research Paper</th>
      <th align="center" width="15%">Datasets</th>
      <th align="center" width="15%">Metrics</th>
      <th align="center" width="15%">Objects</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Publisher</th>
      <th align="center" width="10%">Year</th>
    </tr>  
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_ImVoteNet_Boosting_3D_Object_Detection_in_Point_Clouds_With_Image_CVPR_2020_paper.pdf'> ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes </a></td>
      <td align="left"><ul><li> SUN RGB-D </li></ul></td>
      <td align="left"><ul><li> AP-3D </li></ul></td>
      <td align="left"> 37 object categories</td>
      <td align="left"><a href='https://github.com/facebookresearch/votenet'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection_CVPR_2019_paper.pdf'> Multi-Task Multi-Sensor Fusion for 3D Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-2D </li><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left"> Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/abcSup/NotEnoughSleepAI'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2019</td>   
    </tr>
    
  </tbody>
</table>

## 4. Pseudo-LiDAR

<table>
  <tbody>
    <tr>
      <th width="25%">Research Paper</th>
      <th align="center" width="15%">Datasets</th>
      <th align="center" width="15%">Metrics</th>
      <th align="center" width="15%">Objects</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Publisher</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1906.06310.pdf'> Pseudo-LiDAR++: Accurate Depth for 3D Object Detection in Autonomous Driving </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left"> Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/mileyan/Pseudo_Lidar_V2'> PyTorch </a></td>
      <td align="center">ICLR</td>
      <td align="center">2020</td>   
    </tr>  
    <tr>
      <td><a href='https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Pseudo-LiDAR_From_Visual_Depth_Estimation_Bridging_the_Gap_in_3D_CVPR_2019_paper.pdf'> Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left"> Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='https://github.com/mileyan/pseudo_lidar'> PyTorch </a></td>
      <td align="center">CVPR</td>
      <td align="center">2019</td>   
    </tr>
    
  </tbody>
</table>

## 5. Training tricks

<table>
  <tbody>
    <tr>
      <th width="25%">Research Paper</th>
      <th align="center" width="15%">Datasets</th>
      <th align="center" width="15%">Metrics</th>
      <th align="center" width="15%">Objects</th>
      <th align="center" width="10%">Source Code</th>
      <th align="center" width="10%">Publisher</th>
      <th align="center" width="10%">Year</th>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/2004.00831.pdf'> PPBA: Improving 3D Object Detection through Progressive Population Based Augmentation </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left"> Cars, Pedestrians, Cyclists</td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2020</td>   
    </tr>
    <tr>
      <td><a href='https://arxiv.org/pdf/1908.09492.pdf'> Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection </a></td>
      <td align="left"><ul><li> KITTI </li></ul></td>
      <td align="left"><ul><li> AP-3D </li><li> AP-BEV </li></ul></td>
      <td align="left"> 10 object categories </td>
      <td align="left"><a href='https://github.com/poodarchu/Det3D'> PyTorch </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2019</td>   
    </tr>  
    <tr>
      <td><a href='https://arxiv.org/pdf/1904.04094.pdf'> Weighted Point Cloud Augmentation for Neural Network Training Data Class-Imbalance </a></td>
      <td align="left"><ul><li> ScanNet </li><li> Semantic3D </li></ul></td>
      <td align="left"><ul></ul></td>
      <td align="left"> </td>
      <td align="left"><a href='...'> --- </a></td>
      <td align="center">ArXiv</td>
      <td align="center">2019</td>   
    </tr>
  </tbody>
</table>

## 6. Object tracking (in progress)



**To do list:**
- [x] Add 3D object detection papers based on LiDAR/monocular images/fusion/pseudo-LiDAR.
- [x] Add training tricks papers
- [ ] Add object tracking papers.
- [ ] Provide `download.py` script to automatically download `.pdf` files.


## References

- The format of the README has been referred from [RedditSota/state-of-the-art-result-for-machine-learning-problems](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems)
