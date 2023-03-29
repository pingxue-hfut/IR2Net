# IR2Net

This project is the PyTorch implementation of our paper : IR^2Net: Information Restriction and Information Recovery for Accurate Binary Neural Networks. 
Trained Models of IR2Net-A/B/C are available, the link in the file "Trained Models".

**Dependencies**

- Python 3.7
- Pytorch == 1.3.0

**Accuracy** 

CIFAR-10:

|   Model   | Bit-Width (W/A) | Accuracy (%) |
| --------- | --------------- | ------------ |
| ResNet-20 | 1 / 1           | 87.2         |
| VGG-Small | 1 / 1           | 91.5         |
| ResNet-18 | 1 / 1           | 92.5         | 

ImageNet:

|   Method  | Bit-Width (W/A) | Top-1 (%) | Top-5 (%) |
| --------- | --------------- | --------- | --------- |
| IR2Net-A  | 1 / 1           | 68.2      | 88.0      |
| IR2Net-B  | 1 / 1           | 67.0      | 87.1      |
| IR2Net-C  | 1 / 1           | 66.6      | 87.0      |
| IR2Net-D  | 1 / 1           | 63.8      | 85.5      |

**Code Reference** 

[1] H. Qin, R. Gong, X. Liu, M. Shen, Z. Wei, F. Yu, and J. Song, “Forward and backward information retention for accurate binary neural
networks,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 2247–2256.

[2] Z. Liu, Z. Shen, M. Savvides, and K. Cheng, “Reactnet: Towards precise binary neural network with generalized activation functions,” in Proc.
European Conference on Computer Vision, 2020, pp. 143–159.

[3] K. Han, Y. Wang, Q. Tian, J. Guo, C. Xu, and C. Xu, “Ghostnet: More features from cheap operations,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2020, pp. 1577–1586.

## Citation

If you find our code useful for your research, please consider citing:

    @article{xue2023IR^2Net,
      title={IR^2Net: information restriction and information recovery for accurate binary neural networks},
      DOI={10.1007/s00521-023-08495-z},
      author={Xue, Ping and Lu, Yang and Chang, Jingfei and Wei, Xing and Wei, Zhen},
      journal={Neural Computing and Applications},
      year={2023},
      month={Mar}
    }

