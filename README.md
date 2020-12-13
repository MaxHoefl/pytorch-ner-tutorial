# Tutorial on NER with Pytorch

Following [the stanford tutorial](https://cs230.stanford.edu/blog/namedentity/) on NER.
The corresponding project on Github is [here](https://github.com/cs230-stanford/cs230-code-examples)


## Setup

- Install pipenv: `pip install pipenv`  
- Install requirements: `pipenv install`   
- Copy `template.env` into `.env` and fill variables

| Variable | Explanation | 
| ------------- |:-------------:|
| `KAGGLE_USERNAME` | Username used in API token for Kaggle |
| `KAGGLE_KEY` | Key used in API token for Kaggle |

- Download data: `kaggle datasets download abhinavwalia95/entity-annotated-corpus`   
- Unzip data: `mkdir data && unzip entity-annotated-corpus -d data/`  


## Creating dataset

- Create dataset `python src/build_kaggle_dataset.py`  
- Create vocab `python build_vocab.py --data_dir data/`

