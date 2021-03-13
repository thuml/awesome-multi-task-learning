# Awesome Multi-Task Learning

By [Jialong Wu](https://github.com/Manchery).

I greatly appreciate those surveys below, which help me a lot. *Please let me know if you find any mistakes or omissions!*

## Table of Contents

- [Survey](#survey)
- [Benchmarks](#benchmarks)
  - [Computer Vision](#computer-vision)
  - [NLP](#nlp)
  - [RL & Robotics](#rl--robotics)
- [Architecture](#architecture)
  - [Hard Parameter Sharing](#hard-parameter-sharing)
  - [Soft Parameter Sharing](#soft-parameter-sharing)
  - [Modulation](#modulation)
  - [Decoder-focused Model](#decoder-focused-model)
  - [Routing & NAS](#routing--nas)
  - [Others](#others)
- [Optimization](#optimization)
- [Task Relationship Learning & Grouping](#task-relationship-learning--grouping)
- [Theory](#theory)
- [Misc](#misc)



## Survey

- Vandenhende, S., Georgoulis, S., Proesmans, M., Dai, D., & Van Gool, L.  [Multi-Task Learning for Dense Prediction Tasks: A Survey](https://arxiv.org/abs/2004.13379). TPAMI, 2021.
- Crawshaw, M.  [Multi-Task Learning with Deep Neural Networks: A Survey](http://arxiv.org/abs/2009.09796). ArXiv, 2020. 
- Gong, T., Lee, T., Stephenson, C., Renduchintala, V., Padhy, S., Ndirango, A., Keskin, G., & Elibol, O. H.  [A Comparison of Loss Weighting Strategies for Multi task Learning in Deep Neural Networks](https://ieeexplore.ieee.org/document/8848395). 2019.
- Li, J., Liu, X., Yin, W., Yang, M., Ma, L., & Jin, Y.  [Empirical Evaluation of Multi-task Learning in Deep Neural Networks for Natural Language Processing](http://arxiv.org/abs/1908.07820). ArXiv, 2019.
- Ruder, S.  [An Overview of Multi-Task Learning in Deep Neural Networks](http://arxiv.org/abs/1706.05098). ArXiv, 2017. 
- Zhang, Y., & Yang, Q.  [A Survey on Multi-Task Learning](http://arxiv.org/abs/1707.08114). ArXiv, 2017.



## Benchmarks

### Computer Vision

- NYUv2 [[URL](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)]
  - 3 Tasks: Semantic Segmentation, Depth Estimation, Surface Normal Estimation
  - Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. (2012). [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf). ECCV, 2012.
- CityScapes [[URL](https://www.cityscapes-dataset.com/)]
  - 3 Tasks: Semantic Segmentation, Instance Segmentation, Depth Estimation
- PASCAL Context [[URL](https://cs.stanford.edu/~roozbeh/pascal-context/)]
  - Tasks: Semantic Segmentation, Human Part Segmentation, Semantic Edge Detection, Surface Normals Prediction, Saliency Detection.
- CelebA [[URL](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)]
  - Tasks: 40 human face Attributes.
- Taskonomy [[URL](http://taskonomy.stanford.edu/)]
  - 26 Tasks: Scene Categorization, Semantic Segmentation, Edge Detection, Monocular Depth Estimation, Keypoint Detection, *etc*.
  - Zamir, A. R., Sax, A., Shen, W., Guibas, L. J., Malik, J., & Savarese, S.  [Taskonomy: Disentangling Task Transfer Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.pdf). CVPR, 2018.
- Visual Domain Decathlon [[URL](https://www.robots.ox.ac.uk/~vgg/decathlon/)]
  - 10 Datasets: ImageNet, Aircraft, CIFAR100, *etc*. 
  - Rebuffi, S.-A., Bilen, H., & Vedaldi, A.  [Learning multiple visual domains with residual adapters](https://arxiv.org/abs/1705.08045). NeurIPS, 2017.

### NLP

:TODO:

### RL & Robotics

:TODO:



## Architecture

### Hard Parameter Sharing

- Caruana, R. [Multitask Learning](https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf). 1997.

<p align="center">
  <img src="imgs/hard-parameter-sharing.png" alt="client-demo" width="600px" />
</p>

### Soft Parameter Sharing

- Ruder, S., Bingel, J., Augenstein, I., & Søgaard, A.  [Latent Multi-task Architecture Learning](https://arxiv.org/abs/1705.08142). AAAI, 2019.
- Gao, Y., Ma, J., Zhao, M., Liu, W., & Yuille, A. L.  [NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction](https://arxiv.org/abs/1801.08297). CVPR, 2019.
- Misra, I., Shrivastava, A., Gupta, A., & Hebert, M.  [Cross-Stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539). CVPR, 2016.
- Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., & Hadsell, R.  [Progressive Neural Networks](https://arxiv.org/abs/1606.04671). ArXiv, 2016.
- Yang, Y., & Hospedales, T. [Deep Multi-task Representation Learning: A Tensor Factorisation Approach](https://arxiv.org/abs/1605.06391). ArXiv, 2016.
- Yang, Y., & Hospedales, T. M. [Trace Norm Regularised Deep Multi-Task Learning](http://arxiv.org/abs/1606.04038). ArXiv, 2016.

### Modulation

- Liu, S., Johns, E., & Davison, A. J.  [End-to-End Multi-Task Learning with Attention](http://arxiv.org/abs/1803.10704). CVPR, 2019. 
- Strezoski, G., Noord, N., & Worring, M.  [Many Task Learning With Task Routing](https://arxiv.org/abs/1903.12117). ICCV, 2019.
- Maninis, K.-K., Radosavovic, I., & Kokkinos, I.  [Attentive Single-Tasking of Multiple Tasks](http://arxiv.org/abs/1904.08918). CVPR, 2019.
- Zhao, X., Li, H., Shen, X., Liang, X., & Wu, Y.  [A Modulation Module for Multi-task Learning with Applications in Image Retrieval](https://arxiv.org/abs/1807.06708). ECCV, 2018.
- Rebuffi, S.-A., Vedaldi, A., & Bilen, H.  [Efficient Parametrization of Multi-domain Deep Neural Networks](https://arxiv.org/abs/1803.10082). CVPR, 2018.
- Rebuffi, S.-A., Bilen, H., & Vedaldi, A.  [Learning multiple visual domains with residual adapters](https://arxiv.org/abs/1705.08045). NeurIPS, 2017.

### Decoder-focused Model

- Vandenhende, S., Georgoulis, S., & Van Gool, L.  [MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning](http://arxiv.org/abs/2001.06902). ECCV, 2020.
- Zhang, Z., Cui, Z., Xu, C., Yan, Y., Sebe, N., & Yang, J.  [Pattern-Affinitive Propagation Across Depth, Surface Normal and Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.pdf). CVPR, 2019. 
- Xu, D., Ouyang, W., Wang, X., & Sebe, N.  [PAD-Net: Multi-tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing](https://arxiv.org/abs/1805.04409). CVPR, 2018.

### Routing & NAS

- Sun, X., Panda, R., & Feris, R.  [AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning](http://arxiv.org/abs/1911.12423). NeurIPS, 2020. 
- Bruggemann, D., Kanakis, M., Georgoulis, S., & Van Gool, L.  [Automated Search for Resource-Efficient Branched Multi-Task Networks](http://arxiv.org/abs/2008.10292). BMVC, 2020. 
- Gao, Y., Bai, H., Jie, Z., Ma, J., Jia, K., & Liu, W. [MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning](https://arxiv.org/abs/2003.14058). CVPR, 2020.
- Bragman, F., Tanno, R., Ourselin, S., Alexander, D., & Cardoso, J.  [Stochastic Filter Groups for Multi-Task CNNs: Learning Specialist and Generalist Convolution Kernels](https://arxiv.org/abs/1908.09597). ICCV, 2019.
- Ma, J., Zhao, Z., Chen, J., Li, A., Hong, L., & Chi, E. H.  [SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning](https://ojs.aaai.org/index.php/AAAI/article/view/3788/3666). AAAI, 2019.
- Maziarz, K., Kokiopoulou, E., Gesmundo, A., Sbaiz, L., Bartok, G., & Berent, J.  [Gumbel-Matrix Routing for Flexible Multi-task Learning](http://arxiv.org/abs/1910.04915). ArXiv,  2019.
- Newell, A., Jiang, L., Wang, C., Li, L.-J., & Deng, J.  [Feature Partitioning for Efficient Multi-Task Architectures](https://arxiv.org/abs/1908.04339). ArXiv, 2019.
- Rosenbaum, C., Klinger, T., & Riemer, M.  [Routing Networks: Adaptive Selection of Non-linear Functions for Multi-Task Learning](http://arxiv.org/abs/1711.01239). ICLR, 2018.
- Meyerson, E., & Miikkulainen, R.  [Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering](http://arxiv.org/abs/1711.00108). ICLR, 2018.
- Fernando, C., Banarse, D., Blundell, C., Zwols, Y., Ha, D., Rusu, A. A., Pritzel, A., & Wierstra, D.  [PathNet: Evolution Channels Gradient Descent in Super Neural Networks](http://arxiv.org/abs/1701.08734). ArXiv, 2017. 

### Others

- Zamir, A., Sax, A., Yeo, T., Kar, O., Cheerla, N., Suri, R., Cao, Z., Malik, J., & Guibas, L.  [Robust Learning Through Cross-Task Consistency](http://arxiv.org/abs/2006.04096). CVPR, 2020. 
- Sun, T., Shao, Y., Li, X., Liu, P., Yan, H., Qiu, X., & Huang, X.  [Learning Sparse Sharing Architectures for Multiple Tasks](http://arxiv.org/abs/1911.05034). AAAI, 2020.
- Lee, H. B., Yang, E., & Hwang, S. J.  [Deep Asymmetric Multi-task Feature Learning](http://proceedings.mlr.press/v80/lee18d/lee18d.pdf). ICML, 2018.
- Zhang, Y., Wei, Y., & Yang, Q.  [Learning to Multitask](https://papers.nips.cc/paper/2018/file/aeefb050911334869a7a5d9e4d0e1689-Paper.pdf). NeurIPS, 2018.
- Mallya, A., Davis, D., & Lazebnik, S.  [Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights](https://arxiv.org/abs/1801.06519). ECCV 2018.
- Mallya, A., & Lazebnik, S.  [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/abs/1711.05769). CVPR, 2018.
- Kokkinos, I.  [UberNet: Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision Using Diverse Datasets and Limited Memory](https://arxiv.org/abs/1609.02132). CVPR, 2017.
- Long, M., Cao, Z., Wang, J., & Yu, P. S.  [Learning Multiple Tasks with Multilinear Relationship Networks](https://proceedings.neurips.cc/paper/2017/file/03e0704b5690a2dee1861dc3ad3316c9-Paper.pdf). NeurIPS, 2017.
- Lee, G., Yang, E., & Hwang, S. J.  [Asymmetric Multi-task Learning based on Task Relatedness and Confidence](http://proceedings.mlr.press/v48/leeb16.pdf). ICML, 2016.



## Optimization

- Liu, L., Li, Y., Kuang, Z., Xue, J.-H., Chen, Y., Yang, W., Liao, Q., & Zhang, W.  [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr). ICLR, 2021.
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C.  [Gradient Surgery for Multi-Task Learning](http://arxiv.org/abs/2001.06782). NeurIPS, 2020.
- Ma, P., Du, T., & Matusik, W. [Effcient Continuous Pareto Exploration in Multi-Task Learning](https://arxiv.org/abs/2006.16434). ICML, 2020.
- Lin, X., Zhen, H.-L., Li, Z., Zhang, Q.-F., & Kwong, S.  [Pareto Multi-Task Learning](http://papers.nips.cc/paper/9374-pareto-multi-task-learning.pdf). NeurIPS, 2019.
- Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A.  [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](http://arxiv.org/abs/1711.02257). ICML, 2018.
- Guo, M., Haque, A., Huang, D.-A., Yeung, S., & Fei-Fei, L.  [Dynamic Task Prioritization for Multitask Learning](https://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Focus_on_the_ECCV_2018_paper.pdf). ECCV, 2018.
- Kendall, A., Gal, Y., & Cipolla, R.  [Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115). CVPR, 2018.
- Sener, O., & Koltun, V.  [Multi-Task Learning as Multi-Objective Optimization](http://arxiv.org/abs/1810.04650). NeurIPS, 2018.
- Zhang, Z., Luo, P., Loy, C. C., & Tang, X. [Facial Landmark Detection by Deep Multi-task Learning](https://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf). ECCV, 2014.



## Task Relationship Learning & Grouping

- Vandenhende, S., Georgoulis, S., De Brabandere, B., & Van Gool, L.  [Branched Multi-Task Networks: Deciding What Layers To Share](http://arxiv.org/abs/1904.02920). BMVC, 2020.
- Standley, T., Zamir, A. R., Chen, D., Guibas, L., Malik, J., & Savarese, S.  [Which Tasks Should Be Learned Together in Multi-task Learning?](http://arxiv.org/abs/1905.07553) ICML, 2020.
- Guo, P., Lee, C.-Y., & Ulbricht, D.  [Learning to Branch for Multi-Task Learning](https://arxiv.org/abs/2006.01895). ICML, 2020.
- Dwivedi, K., & Roig, G.  [Representation Similarity Analysis for Efficient Task Taxonomy & Transfer Learning](https://arxiv.org/abs/1904.11740). CVPR, 2019.
- Guo, H., Pasunuru, R., & Bansal, M. [AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning](https://arxiv.org/abs/1904.04153). NAACL, 2019.
- Alonso, H. M., & Plank, B.  [When is multitask learning effective? Semantic sequence prediction under varying data conditions](http://arxiv.org/abs/1612.02251). EACL, 2017. 
- Bingel, J., & Søgaard, A.  [Identifying beneficial task relations for multi-task learning in deep neural networks](http://arxiv.org/abs/1702.08303). EACL, 2017.
- Hand, E. M., & Chellappa, R.  [Attributes for Improved Attributes: A Multi-Task Network Utilizing Implicit and Explicit Relationships for Facial Attribute Classiﬁcation](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14749/14282). AAAI, 2017.
- Lu, Y., Kumar, A., Zhai, S., Cheng, Y., Javidi, T., & Feris, R.  [Fully-adaptive Feature Sharing in Multi-Task Networks with Applications in Person Attribute Classification](http://arxiv.org/abs/1611.05377). CVPR, 2017.
- Kang, Z., Grauman, K., & Sha, F.  [Learning with Whom to Share in Multi-task Feature Learning](http://www.cs.utexas.edu/~grauman/papers/icml2011.pdf). ICML, 2011.
- Kumar, A., & Daume III, H. [Learning Task Grouping and Overlap in Multi-task Learning](http://arxiv.org/abs/1206.6417). ICML, 2012.



## Theory

- Tripuraneni, N., Jordan, M. I., & Jin, C.  [On the Theory of Transfer Learning: The Importance of Task Diversity](https://proceedings.neurips.cc//paper/2020/file/59587bffec1c7846f3e34230141556ae-Paper.pdf). NeurIPS, 2020.
- Wu, S., Zhang, H. R., & Re, C.  [Understanding and Improving Information Transfer in Multi-Task Learning](https://arxiv.org/abs/2005.00944). ICLR, 2020.



## Misc

- Lu, J., Goswami, V., Rohrbach, M., Parikh, D., & Lee, S. [12-in-1: Multi-Task Vision and Language Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.pdf). CVPR, 2020.
- Mao, C., Gupta, A., Nitin, V., Ray, B., Song, S., Yang, J., & Vondrick, C. [Multitask Learning Strengthens Adversarial Robustness](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470154.pdf). ECCV, 2020.
- Guo, P., Xu, Y., Lin, B., & Zhang, Y.  [Multi-Task Adversarial Attack](http://arxiv.org/abs/2011.09824). ArXiv, 2020.
- Zimin, A., & Lampert, C. H. [Tasks Without Borders: A New Approach to Online Multi-Task Learning](https://openreview.net/pdf?id=HkllV5Bs24). AMTL Workshop at ICML 2019.
- Doersch, C., & Zisserman, A.  [Multi-task Self-Supervised Visual Learning](http://arxiv.org/abs/1708.07860). ICCV, 2017.
- Smith, V., Chiang, C.-K., Sanjabi, M., & Talwalkar, A. S.  [Federated Multi-Task Learning](https://proceedings.neurips.cc/paper/2017/file/6211080fa89981f66b1a0c9d55c61d0f-Paper.pdf). NeurIPS, 2017.
- Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N., Jones, L., & Uszkoreit, J.  [One Model To Learn Them All](http://arxiv.org/abs/1706.05137). ArXiv, 2017. 
- Yang, Y., & Hospedales, T. M. [A Unified Perspective on Multi-Domain and Multi-Task Learning](http://arxiv.org/abs/1412.7489). ICLR, 2015.