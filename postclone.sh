#!/bin/bash

repodir=$(pwd)

## Vim installation
function install_vim {
    mkdir -p ~/tmp/
    cd ~/tmp/
    if [ ! -d vimrc/ ]; then
        git clone https://github.com/MaxHoefl/vimrc.git
    fi
    cd vimrc/
    ./install.sh
    cd $repodir
}

## Python dependency installation
function setup_pipenv {
    pip install pipenv
    pipenv install
}

## Downloading data required for NLP model
function download_dataset {
    mkdir -p data/
    echo "Enter kaggle API credentials: "
    read -p "username: " username
    read -p "password: " password
    export KAGGLE_USERNAME=$username
    export KAGGLE_KEY=$password
    pipenv run kaggle datasets download abhinavwalia95/entity-annotated-corpus
    unzip entity-annotated-corpus -d data/
    rm entity-annotated-corpus.zip
}

## Preprocessing data
function process_dataset {
    pipenv run python src/build_kaggle_dataset.py
    pipenv run python src/build_vocab.py --data_dir data/
}

install_vim
setup_pipenv
download_dataset
process_dataset
