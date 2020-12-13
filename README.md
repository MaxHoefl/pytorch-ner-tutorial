# Tutorial on NER with Pytorch

Following [the stanford tutorial](https://cs230.stanford.edu/blog/namedentity/) on NER.


## Setup

- Install requirements: `pipenv install`   
- Copy `template.env` into `.env` and fill variables

| Variable | Explanation | 
| ------------- |:-------------:|
| `KAGGLE_USERNAME` | Username used in API token for Kaggle |
| `KAGGLE_KEY` | Key used in API token for Kaggle |

- Download data: `kaggle datasets download abhinavwalia95/entity-annotated-corpus`   
- Unzip data: `mkdir data && unzip entity-annotated-corpus -d data/`  
