# Sub-Clustering for Class Distance Recalculation in Long-Tailed Drug Discovery

### Overview

The long-tailed distribution commonly seen in AI-aided drug discovery poses significant challenges for employment of machine learning models. Previous approaches, such as resampling and reweighting, attempt to address this by balancing head and tail classes. However, these methods often overemphasize tail classes, though some of which may be easier to identify. To address this issue, we propose a sub-clustering contrastive learning method which clusters the head class into several subclasses, with each subclass containing the similar number of samples as a tail class. Then, we incorporate subclass distance information into the loss function to balance class representation. Additionally, we evaluate the spacing of original classes and the subclasses formed after clustering to assess recognition difficulty, adjusting the loss weight accordingly. A dynamic process is also used to keep the clustering and class spacing updated during training. Our method effectively balances sample instances, class distribution, and classification difficulty. We validate the state-of-the-art performance of our method through extensive experiments on multiple existing long-tailed drug datasets.


### Installation

#### Using `conda`

```bash
conda env create -f environment.yml
conda activate LTDD
```

### Dataset

In this paper, we use three long-tailed dataset (HIV, SBAP, USPTO50k), and provide two different split method ( random split, standard split).

The initial data is SMILES-based. The save path is "./data/dataset name/dataset name.tab". Example: "./data/HIV/hiv.tab".

With dgl library, we transform the data into graph-based data, then spliting it into train set validate set and test set. The save path is "./data/dataset name/split method/dataset name_split method_mode.bin". Example: "./data/HIV/random/HIV_random_train.bin".


### Running Examples

```bash
python main.py --dataset SBAP --split random
python main.py --dataset HIV --split standard
python main.py --dataset USPTO50k --split standrad
```
