#!/bin/bash

SCRIPT_PATH=$(realpath -s "${BASH_SOURCE[-1]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
VENV_DIR="$SCRIPT_DIR/venv"
VENV_ACTIVATE_EXECUTE="$VENV_DIR/bin/activate"

USE_CONDA="0"
CONDA_ENV="py311-vanilla-temp"

conda_deactivate() {
    if [ "$USE_CONDA" == "1" ]; then
        conda deactivate
        yes | conda remove -n $CONDA_ENV --all
    fi
}

create_env() {
    virtualenv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Unexpected error while creating virtual environment."
        conda_deactivate
        return 1
    fi

    if [ "$1" == "--minimal" ]; then
        yes | pip install -r requirements.minimal.txt
    else
        yes | pip install -r requirements.txt
    fi

    if [ $? -ne 0 ]; then
        echo "Unexpected error while installing required python packages."
        conda_deactivate
        return 1
    fi

    printf "\n# set required environment variables for the project" >> "$VENV_ACTIVATE_EXECUTE"
    printf "\nPATH=$SCRIPT_DIR:\$PATH" >> "$VENV_ACTIVATE_EXECUTE"
    printf "\nexport PATH\n" >> "$VENV_ACTIVATE_EXECUTE"
}

echo "Check conda version."
conda --version
if [ $? -eq 0 ]; then
    echo "Create virtual environment of Python 3.11 using Conda."
    yes | conda create -n $CONDA_ENV python=3.11 virtualenv
    conda activate $CONDA_ENV
    USE_CONDA="1"
fi

create_env

source "$VENV_ACTIVATE_EXECUTE"
if [ $? -ne 0 ]; then
    echo "Unexpected error while activating the virtual environment."
    conda_deactivate
    return 1
fi

pip install kaggle

COMPETITION="ml-olympiad-tfugsurabaya-2024"
kaggle competitions download -c $COMPETITION
if [ $? -ne 0 ]; then
    echo "Unexpected error while downloading the competition data."
    conda_deactivate
    deactivate
    return 1
fi

unzip -o $COMPETITION.zip -d data/$COMPETITION
rm $COMPETITION.zip
conda_deactivate
deactivate
