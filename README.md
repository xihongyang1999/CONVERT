[stars-img]: https://img.shields.io/github/stars/xihongyang1999/CONVERT?color=yellow
[stars-url]: https://github.com/xihongyang1999/CONVERT/stargazers
[fork-img]: https://img.shields.io/github/forks/xihongyang1999/CONVERT?color=lightblue&label=fork
[fork-url]: https://github.com/xihongyang1999/CONVERT/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=xihongyang.1999.CONVERT/
[adgc-url]: https://github.com/xihongyang1999/CONVERT

# CONVERT:Contrastive Graph Clustering with Reliable Augmentation

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://www.acmmm2023.org" alt="Conference">
        <img src="https://img.shields.io/badge/ACM MM'23-brightgreen" /></a>
<p/>



[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]


An official source code for paper CONVERT:Contrastive Graph Clustering with Reliable Augmentation, accepted by ACM MM 23. Any communications or issues are welcomed. Please contact xihong_edu@163.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

-------------

### Overview

<p align = "justify"> 
 Illustration of CONVERT:Contrastive Graph Clustering with Reliable Augmentation mechanism. 
</p>
<div  align="center">    
    <img src="./assets/convert.jpg" width=60%/>
</div>







### Requirements

The proposed CONVERT is implemented with python 3.8.8 on a NVIDIA 2080 Ti GPU. 

Python package information is summarized in **requirements.txt**:

- torch==1.8.0
- tqdm==4.61.2
- numpy==1.21.0
- tensorboard==2.8.0



### Quick Start

```
python train.py 
```



### Citation

If you use code or datasets in this repository for your research, please cite our paper.

```
@inproceedings{CONVERT,
  title={CONVERT: Contrastive Graph Clustering with Reliable Augmentation},
  author={Yang, Xihong and Tan, Cheng and Liu, Yue and Liang, Ke and Wang, Siwei and Zhou, Sihang and Xia, Jun and Li, Stan Z and Liu, Xinwang and Zhu, En},
  booktitle={Proceedings of the 31th ACM International Conference on Multimedia},
  pages={},
  year={2023}
}
```



