# ICDAR 2023: 篡改文本检测 赛道一 AI-LORD队伍15名方案介绍
# 赛题介绍
- 赛题地址：[ICDAR 2023: 篡改文本检测]([ICDAR 2023 DTT in Images 1: Text Manipulation Classification_算法大赛_天池大赛-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/532048/introduction))
- 赛题任务：利用模拟电子商务场景的图像篡改文本（TTI），检测文本是否被篡改


# 环境配置
- 操作系统版本：Ubuntu 7.2.0-8ubuntu3.1
- Python 版本：3.8.0
- PyTorch 版本：1.11.0+cu113
- CUDA 版本：11.3
- 其他Python依赖：
```text
numpy==1.23.5
pandas==1.5.3
scikit_learn==1.2.1
torch==1.11.0+cu113
tqdm==4.64.1
albumentations==1.3.0
opencv_python==4.7.0
timm==0.6.12
torchvision==0.14.1
```
# 代码结构
```text
├── code
│   ├── train			# 不同模型的训练
│	│	├── 4-mvss
│	│	├── 1-eff3-768-ygh.py
│	│	├── 2-eff3-512-lj.py
│	│	├── 3-eff7-512.py
│	│	├── 5-eff4-512.py
│	│	├── 6-den-512.py
│	│	├── 7-rex-512.py
│   ├── test
│   ├── run.sh			# 推理脚本
│   ├── train.sh
├── data    
│   ├── session1            # 初赛数据集
├── expand_data    
│   ├── paper            # 扩充数据集
├── user_data    
│   ├── model data
│	│	├── 1mo
│	│	├── 2ef3
│	│	├── 3ef7
│	│	├── 4mv
│	│	├── 5ef4
│	│	├── 6den
│	│	├── 7rex 
├── prediction_result   
├── README.md
├── requirements.txt        # python环境依赖

```


# 比赛策略

- 本次比赛使用了efficientnet-b3网络采用多阶段多尺度训练，加载IMAGENET预训练权重、384分辨率训练，然后加载384得到的权重，再用512训练。之后768测试仍然是efficientnet-b3，在上述512训练后，继续768微调，用768测试。
- 其中，在512训练后加入额外数据(add_data)训练，数据来源于2022篡改比赛数据、爬虫爬取网络图片和参考论文数据。
- 训练阶段采用5折交叉验证，并且在512阶段及之后增加额外数据进行训练。损失函数用FOCAL-loss，优化器ADAM，带有重启的余弦退火策略。
- 此外使用了efficientnet-b7、densenet169、resnext50、mvssnet网络直接加载IMAGENET预训练权重，在512大小源数据加新数据训练训练策略同上。
- 其中MVSSNET网络训练参考https://github.com/dddb11/MVSS，全部数据用来训练，使用ADAM优化器，STEPLR策略等。
- 最后，对这些网络结果加权求和，此时初赛分数可达83.83。


# 引用
- Chen, Xinru, et al. "Image manipulation detection by multi-view multi-scale supervision." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2021.

- 王裕鑫, 张博强, 谢洪涛, 张勇东. 基于空域与频域关系建模的篡改文本图像检测. *网络与信息安全学报*[J], 2022, 8(3): 29-40 doi:10.11959/j.issn.2096-109x.2022035

- https://github.com/dddb11/MVSS

  

