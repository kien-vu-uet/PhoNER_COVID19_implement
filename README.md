# PhoNER_COVID19_implement

## Implementation of paper: [COVID-19 Named Entity Recognition for Vietnamese](https://aclanthology.org/2021.naacl-main.173/)

## Dataset: [`COVID-19 Named Entity Recognition for Vietnamese`](https://github.com/VinAIResearch/PhoNER_COVID19)
# Setup
## Installation

**Note**:
The current version is only compatible with python>=3.6 and jdk11
```bash
git clone https://github.com/kien-vu-uet/PhoNER_COVID19_implement.git
cd PhoNER_COVID19_implement
pip install -r code/requirements.txt
```

## Training
```bash
python code/phobert_fine_tuning.py
```

## Infer
```bash
python code/phobert_infer.py
```

## Visualize results
See images on [`heatmap`](https://github.com/kien-vu-uet/PhoNER_COVID19_implement/tree/main/heatmap)

