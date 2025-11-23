<div align="center">
    <img height=200 src="./.github/images/news-logo.png" alt="News Contents on Smartphone">
</div>

<div align="center"> <img height=200 src="./.github/images/news-logo.png" alt="News contents on smartphone"> </div> <h1 align="center">Assignment 2 - News Recommender System Phase 0 Evaluation</h1> <p align="center"><strong>Inspired by "LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation"</strong></p>
<p align="center"><strong>By: Aanisha Newaz</strong></p>

## Overview

This repository contains the code and experiment setup for Assignment 2, where Phase 0 of the SIGIR 2025 paper:

Hu et al. (2025). “LLM-Generated Fake News Induces Truth Decay in News Ecosystem.” is replicated using:
- The NRMS (Neural News Recommendation with Multi-Head Self-Attention) model
- The original GossipCop FakeNewsNet dataset, converted into MIND-style format
- Synthetic user–news interactions
- Human-written real and fake news only (Phase 0)

Phase 0 examines how a recommendation model ranks human-written fake (HF) vs human-written real (HR) articles, without injecting any LLM-generated content.

This project does not include Phases 1–3 from the paper (LLM-generated articles), nor does it replicate LSTUR.

## Project Structure

The project structure is as below.

```bash
$ project -L 2
├── README.md
├── dataset/
│   ├── gossipcop_raw/
│   └── gossipcop/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── convert_gossipcop_csv.py
│   ├── compute_hf_hr_table.py
├── src/
│   ├── config/
│   ├── const/
│   ├── evaluation/
│   ├── experiment/
│   ├── mind/
│   ├── recommendation/
│   └── utils/
└── output/
```

## Preparation

### Prerequisites

- [Rye](https://rye-up.com/)
- Python 3.11.3
- PyTorch 2.0.1
- transformers 4.30.2

### Environment Setup

At first, create python virtualenv & install dependencies by running

```
$ rye sync
```

If you successfully created a virtual environment, a `.venv/` folder should be created at the project root.

Then, please set `PYTHONPATH` by runnning

```
$ export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Download Microsoft News Dataset (MIND) for training

We use **[MIND (Microsoft News Dataset)](https://msnews.github.io/)** dataset for training and validating the news recommendation model. You can download them by executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py).

```
$ rye run python ./dataset/download_mind.py
```

By executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py), the MIND dataset will be downloaded from an external site and then extracted.

If you successfully executed, `dataset` folder will be structured as follows:

```
./dataset/
├── download_mind.py
└── mind
    ├── large
    │   ├── test
    │   ├── train
    │   └── val
    ├── small
    │   ├── train
    │   └── val
    └── zip
        ├── MINDlarge_dev.zip
        ├── MINDlarge_test.zip
        ├── MINDlarge_train.zip
        ├── MINDsmall_dev.zip
        └── MINDsmall_train.zip
```

### Dataset for Phase 0

Since the dataset used in the SIGIR 2025 study is not publicly available, this project uses:

**GossipCop (FakeNewsNet) from Kaggle**, converted into MIND-style news + behavior format.

Converting GossipCop to MIND-style

Run:

```
rye run python scripts/convert_gossipcop_csv.py
```
This produces:
```
dataset/gossipcop/train/news.tsv
dataset/gossipcop/train/behaviors.tsv
dataset/gossipcop/val/news.tsv
dataset/gossipcop/val/behaviors.tsv
```
Behavior rows are synthetically generated and include:
- user_id
- timestamp
- clicked news
- negative samples
- candidate lists

This matches the expectations of NRMS.


## Training

### Fine Tune a model

Example training command:

```bash
$ rye run python src/experiment/train_full.py -m \
    batch_size=2 \
    npratio=2 \
    history_size=20 \
    max_len=20 \
    epochs=1 \
    gradient_accumulation_steps=1
```

Training was done on:
- Approximately 500 synthetic behavior rows
- DistilBERT as encoder
- 1 epoch due to compute constraints

Validation Split
- Converted GossipCop validation set
- ~2,200 behavior rows
- 3 candidates per impression

### Model Performance
To compute HF and HR metrics:
```
rye run python scripts/compute_hf_hr_table.py
```

#### Experimental Results for Experimental Phase 0 - Trained on 500 Behaviours

|         Model          |  MRR  |  nDCG@5  | nDCG@10 | Ratio@5 | Ratio@10 |
| :--------------------: | :---: | :---: | :----: | :-----: | :-----------: |
| HF  | 40.79 | 48.86 | 55.03  |  63.69  |       63.53       |
| HR | 40.17 | 48.75 | 54.57  |  80.39  |    80.02     |
|    RRA    | -1.51% | -0.24% | -0.83%  |  26.23%  |    25.95%     |

#### Experimental Results for Experimental Phase 0 - Trained on 2000 Behaviours

|         Model          |  MRR  |  nDCG@5  | nDCG@10 | Ratio@5 | Ratio@10 |
| :--------------------: | :---: | :---: | :----: | :-----: | :-----------: |
| HF  | 39.19 | 47.63 | 53.79  |  64.18  |       63.53       |
| HR | 41.21 | 49.74 | 55.38  |  80.62  |    80.02     |
|    RRA    | 5.16% | 4.43% | 2.94%  |  25.60%  |    25.95%     |

#### Experimental Results From Original Study Baseline

|         Model          |  MRR  |  nDCG@5  | nDCG@10 | Ratio@5 | Ratio@10 |
| :--------------------: | :---: | :---: | :----: | :-----: | :-----------: |
| HF  | 17.36 | 46.28 | 44.84  |  43.98  |       43.10       |
| HR | 18.62 | 53.72 | 55.17  |  56.02  |    56.90     |
|    RRA    | 7.26% | 16.08% | 23.04%  |  27.38%  |    32.02%     |



## Reference for Model Source Code

```
@misc{
  yuki-yada-news-recommendation-llm,
  author = {Yuki Yada},
  title = {News Recommendation using LLM},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YadaYuki/news-recommendation-llm}}
}
```

## Reference

[1] **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.
https://aclanthology.org/N19-1423

[2] **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**
Sanh, V., Debut, L., Chaumond, J., & Wolf, T.
https://arxiv.org/abs/1910.01108

[3] **Neural News Recommendation with Multi-Head Self-Attention**
Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., & Xie, X.
https://aclanthology.org/D19-1671

[4] **Empowering News Recommendation with Pre-Trained Language Models**
Wu, C., Wu, F., Qi, T., & Huang, Y.
https://doi.org/10.1145/3404835.3463069

[5] **MIND: A Large-scale Dataset for News Recommendation**
Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J., Wu, W., & Zhou, M.
https://aclanthology.org/2020.acl-main.331

[6] Beizhe Hu, Qiang Sheng, Juan Cao, Yang Li, and Danding Wang. 2025. LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25). Association for Computing Machinery, New York, NY, USA, 435–445. https://doi.org/10.1145/3726302.3730027

[7] B, A. N. (2024, March 27). Gossipcop. Kaggle. https://www.kaggle.com/datasets/akshaynarayananb/gossipcop?resource=download 

