# CG2310:基于机器学习的分子动力学势能和力场拟合



## 代码环境

| Package | Version   |
| ------- | --------- |
| Python  | 3.8.17    |
| PyTorch | 1.13.1    |
| CUDA    | 11.7      |
| numpy   | 1.21.2    |
| RDKit   | 2023.03.2 |

## 数据集

### 数据集文件夹存放形式

``` bash
datasets
├── old_train(存放赛题组提供的原始训练集)
│   ├── add_bond_data.ipynb
│   ├── rmd17_benzene.npz
│   └── *.npz
├── old_test(存放赛题组提供的原始测试集)
│   ├── analysis_data.ipynb
│   ├── rmd17_benzene.npz
│   └── rmd17.npz
├── train(存放预处理后带化学键信息的训练集)
│   ├── check_data.ipynb
│   ├── rmd17_benzene_new.npz
│   └── *_new.npz
├── test(存放预处理后带化学键信息的测试集,共10个)
│   ├── concat_label.ipynb
│   ├── rmd17_benzene_new.npz
│   └── *_new.npz
```

### 数据预处理

分别执行`old_test/analysis.ipynb`和`old_train/add_bond_data.ipynb`

## 训练

如果需要训练某个基于某个特定数据集的模型可以输入如下命令：

```bash
python main.py --dataset $dataset_name --hidden_dim 256 --lr 3e-4
```

若想一次性训练所有10个数据集，可以直接执行一下命令

```bash
bash auto_train.bash
```

## 预测

同训练一样，我们提供了两种方式用于在测试集上预测能量和力，分别是

```bash
python test.py --dataset $dataset_name --hidden_dim 256 --lr 3e-4
```

```bash
bash auto_test.bash
```

预测结束后，请执行`datasets/test`下的`concat_label.ipynb`文件来将所有的预测结果拼接成符合原始测试集`rmd17.npz`的数据组成格式。

## 联系

如果有任何疑问，请联系我 guoth3@mail2.sysu.edu.cn