<img src="media/ICLR-logo.svg" height="40" align="right"/>

# DEXML
Codebase for learning dual-encoder models for (extreme) multi-label classification tasks.

> [Dual-encoders for Extreme Multi-label Classification](https://arxiv.org/pdf/2310.10636v2.pdf) <br>
> Nilesh Gupta, Devvrit Khatri, Ankit S. Rawat, Srinadh Bhojanapalli, Prateek Jain, Inderjit S. Dhillon <br>
> ICLR 2024

## Highlights
- Multi-label retrieval losses DecoupledSoftmax and SoftTopk (replacement for InfoNCE (Softmax) loss in multi-label and top-k retrieval settings)
- Distributed dual-encoder training using gradient caching (allows for a large pool of labels in loss computation without getting OOM)
- State-of-the-art dual-encoder models for extreme multi-label classification benchmarks

## Notebook Demo
See `dexml.ipynb` notebook or try it in this [colab](https://colab.research.google.com/github/nilesh2797/DEXML/blob/main/dexml.ipynb)

## Download pretrained models
| **Dataset** | **P@1** | **P@5** | **HF Model Page** |
|-------------|---------|---------|-------------------|
| **LF-AmazonTitles-1.3M** | 58.40 | 45.46 | https://huggingface.co/quicktensor/dexml_lf-amazontitles-1.3m | 
| **LF-Wikipedia-500K** | 85.78 | 50.53 | https://huggingface.co/quicktensor/dexml_lf-amazontitles-131k | 
| **LF-AmazonTitles-131K** | 42.52 | 20.64 | https://huggingface.co/quicktensor/dexml_lf-amazontitles-131k |
| **EURLex-4K** | 86.78 | 60.19 | https://huggingface.co/quicktensor/dexml_eurlex-4k |

## Training DEXML
### Preparing Data
The codebase assumes following data structure: <br>
<pre>
Datasets/
└── EURLex-4K # Dataset name
    ├── raw
    │   ├── trn_X.txt # train input file, ith line is the text input for ith train data point
    │   ├── tst_X.txt # test input file, ith line is the text input for ith test data point
    │   └── Y.txt # label input file, ith line is the text input for ith label in the dataset
    ├── Y.trn.npz # train relevance matrix (stored in scipy sparse npz format), num_train x num_labels
    └── Y.tst.npz # test relevance matrix (stored in scipy sparse npz format), num_test x num_labels
</pre>
Before running the training/testing the default code expects you to convert the input features to BERT's (or any text transformer) tokenized input indices. You can achieve that by running:
```shell
dataset="EURLex-4K"
python utils/tokenization_utils.py --data-path Datasets/${dataset}/raw/Y.txt --tf-max-len 128 --tf-token-type bert-base-uncased
python utils/tokenization_utils.py --data-path Datasets/${dataset}/raw/trn_X.txt --tf-max-len 128 --tf-token-type bert-base-uncased
python utils/tokenization_utils.py --data-path Datasets/${dataset}/raw/tst_X.txt --tf-max-len 128 --tf-token-type bert-base-uncased
```

For some extreme classification benchmark datasets such as LF-AmazonTitles-131K and LF-AmazonTitles-1.3M, you additionally need test time label filter files (`Datasets/${dataset}/filter_labels_test.txt)`) to get the right results. Please see note on these filter files [here](http://manikvarma.org/downloads/XC/XMLRepository.html#ba-pair) to know more.

### Training commands
Training code assumes all hyperparameter and runtime arguments are specified in a config yaml file. Please see `configs/dual_encoder.yaml` for a brief description of all parameters (you can keep most of the parameters same across experiments). See `configs/EURLex-4K/dist-de-all_decoupled-softmax.yaml` to see some of the important hyperparameters that you may want to change for different experiments.
```shell
# Single GPU
dataset="EURLex-4K"
python train.py configs/${dataset}/dist-de-all_decoupled-softmax.yaml

# Multi GPU
num_gpus=4
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/dist-de-all_decoupled-softmax.yaml
```

## Cite
```bib
@InProceedings{DEXML,
  author    = "Gupta, N. and Khatri, D. and Rawat, A-S. and Bhojanapalli, S. and Jain, P. and Dhillon, I.",
  title     = "Dual-encoders for Extreme Multi-label Classification",
  booktitle = "International Conference on Learning Representations",
  month     = "May",
  year      = "2024"
}
```
