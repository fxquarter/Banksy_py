# BANKSY:  Spatial Clustering Algorithm that Unifies Cell-Typing and Tissue Domain Segmentation
## Overview:本项目是对banksy在visium HD数据集上的复现以及一些改进,通过分析聚类流程调整HD适合参数，融合BPCells实现大规模数据处理，优化空间最近邻图的构建与空间权重的生成几方面进行改进
### 实验所用数据集：Visium_HD_Mouse_Brain，Visium_HD_3prime_Human_Ovarian_Cancer
### 运行环境：python==3.11 anaconda
### 具体改进：融合K-NN建图和半径建图，实现自适应半径剪枝的k-NN空间图，在优化后的空间图基础上，只对经过剪枝的有效边进行计算，同时处于visium hd的高分辨率特性导致内存溢出，采用BPCells框架将矩阵生成占用的部分内存转移至磁盘，降低内存使用门槛。
### 实验效果：
#### nospatial：
![image](https://github.com/user-attachments/assets/feae259b-ec07-4d40-b2de-953f63d537a5)
#### spatial：
![image](https://github.com/user-attachments/assets/1934d688-b31a-477f-863d-a0b8319a5c4f)

#### connection matrix：
![image](https://github.com/user-attachments/assets/87c4bf00-b008-43d9-a13d-ea6f83507966)
