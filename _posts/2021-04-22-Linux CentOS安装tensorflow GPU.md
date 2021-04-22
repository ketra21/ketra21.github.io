---
layout:     post
title:      Linux CentOS安装tensorflow GPU
subtitle:   
date:       2021-04-22
author:     阳光和鱼
header-img: img/post-bg-desk.jpg
catalog: true
tags:

    - 技术经验分享

---

Linux CentOS安装tensorflow GPU

【20210422亲测】【小白友好】

参考了之前的几篇安装教程，有针对tensorflow1.0的内容比较过时，【参考1】的cuda等版本的选择比较繁杂，而cuda的安装有些省略，对小白不友好，不过我还是通过各种找资料最终安装成功，为了方便后来者参考，结合最新的版本，凑成此文。

> 配置：
>
> - CentOS系统
>
> - GPU：GeForce GTX 1080 Ti

## 一、安装前准备工作

### 1、查看当前服务器的显卡

1.1 查看VGA接口，执行：

````
lspci | grep VGA
````

结果：

```
17:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
65:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
```

说明有两个显卡，如果结果中的“00:”前缀（此处分别是17，65），表示该显卡是虚拟机上挂载的显卡。[^1]

如果没有结果，说明没有显卡，只能参考其他教程安装CPU版本的tensorflow。

1.2 查看显卡型号，执行：

```
lspci | grep NVIDIA
```

结果：

```
17:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
17:00.1 Audio device: NVIDIA Corporation GP102 HDMI Audio Controller (rev a1)
65:00.0 VGA compatible controller: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1)
65:00.1 Audio device: NVIDIA Corporation GP102 HDMI Audio Controller (rev a1)
```

1.3 执行：

````
nvcc -V
````

如果没有输出，则说明cuda没有安装。

### 2、禁用nouveau[^1]

- 可选，我验证结果是没有重启机器仍然可以安装。

（1）查看系统自带的驱动：

```
lsmod | grep nouveau
```

如果有结果，则说明存在nouveau。没有则直接跳过这一步。

（2）编辑如下文件：

```
vim /usr/lib/modprobe.d/dist-blacklist.conf
```

然后在最后添加如下内容：

```
blacklist nouveau
options nouveau modeset=0
```

## 二、安装Nvidia驱动

### **1、查看是否有历史安装**

执行如下命令：

```shell
nvidia-smi
```

- 如果没有输出，则说明驱动没有安装。
- 如果有输出，可以查看配置，并跳过此驱动安装， 直接跳到“三、安装cuda和cudnn”

```
Thu Apr 22 20:00:02 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.56       Driver Version: 460.56       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:17:00.0 Off |                  N/A |
| 23%   40C    P5    12W / 250W |      0MiB / 11178MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:65:00.0 Off |                  N/A |
| 28%   40C    P5    21W / 250W |      0MiB / 11175MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 2、 安装基础rpm依赖包[^1]

- 安装Nvidia驱动所需要的依赖包为kernel-devel、gcc、dkms。
- 验证cudn，编译并运行验证Sample代码所需要的依赖包为gcc-c++。

2.1 确认并安装对应服务器内核版本的kernel-devel包。

首先确认当前服务器的内核版本：

```
[root@localhost gpu-software]#  uname -r
3.10.0-862.el7.x86_64
```

然后查看当前yum环境中提供的kernel-devel包的版本：

```
[root@localhost gpu-software]# yum list|grep kernel-devel
kernel-devel.x86_64                     3.10.0-862.el7                 http239
```

yum提供的kernel-devel的版本是不是和当前服务器内核版本一致，如果一致则直接安装，不一致则更换一个和当前内核一致的版本。

2.2 可选：添加一个合适的yum源，因为默认的yum源没有dkms包。参考yum源如下：

```
cat /etc/yum.repos.d/epel.repo

[epel]
name=Extra Packages for Enterprise Linux 7 - $basearch
baseurl=http://mirrors.aliyun.com/epel/7/$basearch
failovermethod=priority
enabled=1
gpgcheck=0
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7

[epel-debuginfo]
name=Extra Packages for Enterprise Linux 7 - $basearch - Debug
baseurl=http://mirrors.aliyun.com/epel/7/$basearch/debug
failovermethod=priority
enabled=0
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7
gpgcheck=0

[epel-source]
name=Extra Packages for Enterprise Linux 7 - $basearch - Source
baseurl=http://mirrors.aliyun.com/epel/7/SRPMS
failovermethod=priority
enabled=0
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7
gpgcheck=0
```

2.3 安装rpm包。

```
[root@localhost ~]# yum install gcc dkms gcc-c++
```

### 3、安装Nvidia驱动

- 官网下载地址为：https://www.nvidia.cn/Download/index.aspx?lang=cn
- 从官网上下载符合当前服务器显卡型号的驱动后，对下载后对文件添加可执行权限，然后执行如下命令：

```
./NVIDIA-Linux-x86_64-418.87.00.run --kernel-source-path=/usr/src/kernels/3.10.0-862.el7.x86_64
```

- 注意如下选项：

> Would you like to register the kernel module sources with DKMS? This will allow DKMS to automatically build a new module, if you install a different kernel later.

这里选择Yes。

- 安装结束后，执行如下命令可以查看显卡信息，则说明安装成功：

```
nvidia-smi
```

## 三、 安装cuda和cudnn

### 1、安装cuda

如果执行：

```
nvcc -V
```

没有输出，继续；否则则跳过此部分。

在Cuda toolkit的下载界面（[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)），选择系统版本和安装方式

![](https://i.niupic.com/images/2021/04/22/9gIn.png)

下载文件。

> 如果不知道Linux的版本信息，可以参考如下tips：
>
> - 查看Linux架构：执行`uname -a`，输出：
>
> ```
> Linux localhost.localdomain 3.10.0-1127.18.2.el7.x86_64 #1 SMP Sun Jul 26 15:27:06 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
> ```
>
> Architecture选择**x86_64**
>
> - 查看Linux版本的方法：执行`cat /etc/redhat-release`，输出
>
> ```
> CentOS Linux release 7.8.2003 (Core)
> ```
>
> Distribution选择**CentOS**，版本Version选**7**，安装方式Installer Type选择**runfile(local)**

执行下载代码（大概3.5G，需要时间较长，注意避免网络中断）：

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
```

执行安装代码（时间较长）：

```shell
sudo sh cuda_11.3.0_465.19.01_linux.run
```

询问是否接受是输入`accept`并回车。

<img src="https://i.niupic.com/images/2021/04/22/9gIw.png" style="zoom:67%;" />

出现安装选项，因为我们已经安装了Driver，如下图中按enter取消driver的选择（无X表示不安装），然后移动光标选择到Install，回车进行安装。

<img src="https://i.niupic.com/images/2021/04/22/9gIv.png" style="zoom:67%;" />

安装好CUDA Toolkit后，屏幕输出：

```
Driver:   Installed
Toolkit:  Installed in /usr/local/cuda-10.2/
Samples:  Installed in /home/abneryepku/

Please make sure that
 -   PATH includes /usr/local/cuda-10.2/
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root
```

执行进入环境变量文件：

```
vim ~/.bashrc
```

在文件最后加入如下内容：

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
```

![](https://i.niupic.com/images/2021/04/22/9gIp.png)

输入`:q`回车，退出编辑界面。

执行：

```shell
source /etc/profile
```

执行：

```
nvcc -V
```

输出如下，则说明安装成功：

```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Tue_Sep_15_19:10:02_PDT_2020
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1.TC455_06.29069683_0
```

### **2、安装cudnn**

**2.1 下载cudnn**

花5分钟在官网：[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)免费注册账号。

然后进入下载界面[https://developer.nvidia.com/cudnn-download-survey](https://developer.nvidia.com/cudnn-download-survey)，选择`cuDNN Library for Linux[x86_64]`，下载到本地，再上传到服务器中。

解压cudnn包，执行：

```
tar cudnn-11.2-linux-x64-v8.1.1.33.tgz
```

在本目录会多一个cuda文件夹。

将cudnn中的文件复制到cuda目录，并添加权限，执行：

```
cp cuda/include/cudnn.h /usr/local/cuda-10.2/include
cp cuda/lib64/libcudnn* /usr/local/cuda-10.2/lib64
chmod a+r /usr/local/cuda-10.2/include/cudnn.h  /usr/local/cuda-10.2/lib64/libcudnn*
```

## 四、安装Tensorflow[^2]

### 1、安装tensorflow，执行：

```shell
pip install --upgrade tensorflow
```

### 2、测试TF，执行：

```
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

如果返回张量，说明安装成功。

# 参考资料

[^1]CentOS 7安装GPU、Cuda、Tensorflow https://www.cnblogs.com/shenggang/p/12133220.html

[^2]使用 pip 安装 TensorFlow https://www.tensorflow.org/install/pip#virtual-environment-install