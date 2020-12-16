#!/bin/bash

pip install pipenv
pipenv install

mkdir -p data/
echo "Enter kaggle API credentials: "
read -p "username: " username
read -p "password: " password
export KAGGLE_USERNAME=$username
export KAGGLE_KEY=$password
pipenv run kaggle datasets download abhinavwalia95/entity-annotated-corpus
unzip entity-annotated-corpus -d data/
rm entity-annotated-corpus.zip
pipenv run python src/build_kaggle_dataset.py
pipenv run python src/build_vocab.py --data_dir data/


