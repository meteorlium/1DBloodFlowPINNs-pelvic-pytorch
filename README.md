# 1DBloodFlowPINNs-pelvic-pytorch

## 简介

本项目将[`PredictiveIntelligenceLab/1DBloodFlowPINNs`](https://github.com/PredictiveIntelligenceLab/1DBloodFlowPINNs)中的Pelvic部分使用pytorch复现。

原项目是论文《Georgios Kissas, Yibo Yang, Eileen Hwuang, Walter R. Witschey, John A. Detre, Paris Perdikaris. "[Machine  learning in cardiovascular flows modeling: Predicting pulse wave  propagation from non-invasive clinical measurements using  physics-informed deep learning.](https://www.sciencedirect.com/science/article/pii/S0045782519305055?dgcid=author)" (2019).》的代码。该论文使用PINN(pysics informed neurol network)神经网络结构求解Navier-Stokes方程，从而计算主动脉中的血压分布。

下面介绍本项目中的代码。

## 代码

### 文件

- /data：用到的数据。
- /pyfile1207：某次训练后的结果，仅供参考。包括result_loss，result_model，result_Pressure，result_Velocity。这些文件夹说明见下方。
- /result_loss：储存训练过程中的loss变化。使用draw_log_loss.py可以画图。
- /result_model：储存训练得到的模型。使用net.py中的predict函数可以调用并测试。
- /result_Pressure：在训练过程，使用当前的模型进行预测，并于有限元计算的结果进行对比。
- /result_Velocity：与上一个类似，分别计算的是pressure和velocity的值。
- dataset.py：将数据制作成pytorch需要的dataset的形式。
- draw_log_loss.py：画出训练过程中loss变化图。
- loss.py：PINN中的定义损失函数。
- net.py：PINN网络结构。
- 

### requirements

- pytorch 1.7,1+cuda101 ，安装：pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
- numpy
- matplotlib
- os
- json

### PINN

网络结构为全连接层，代码为net.py。

损失函数为NS方程残差的均方误差、测量值的均方误差、连接点约束的均方误差之和，详细设置参考论文，代码为loss.py。

损失函数中需要的偏导数使用torch.autograd.grad()计算，代码为train.py中170行前后。

### 训练和测试

训练脚本为train.py。训练参数为，先使用lr=1e-3训练290000次，再使用lr=1e-4训练50000次。训练时间约为44小时，原tensorflow程序运行时间为7小时，时间较长的原因可能是因为本程序每次训练过程中七个网络依次计算，如果使用7个gpu改为并行计算可能可以大大提升训练速度。

测试脚本为test.py，可自定义x，t，i_vessel。

### 模型结果

pyflie1207保存了一次训练得到的

