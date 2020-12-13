# Tutorial on NER with Pytorch

Following [the stanford tutorial](https://cs230.stanford.edu/blog/namedentity/) on NER.


## Setup

- Install requirements: `pipenv install`   
- Upload kaggle API credentials `kaggle.json`(created on kaggle -> Account -> API) to
`~/.kaggle/`. Make sure that permissions are set to `600`.   
- Download data: `kaggle datasets download abhinavwalia95/entity-annotated-corpus`   
- Unzip data: `mkdir data && unzip entity-annotated-corpus -d data/`  
