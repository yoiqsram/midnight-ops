#!/bin/bash

SCRIPT_PATH=$(realpath -s "${BASH_SOURCE[-1]}")
PYTHONPATH="$(dirname "$SCRIPT_PATH")"
COMPETITION="ml-olympiad-tfugsurabaya-2024"

if [ "$1" == "--minimal" ]; then
    conda create --prefix ./venv -f jupyterlab pandas nltk mlflow
elif [ "$1" == "" ]; then
    conda create --prefix ./venv -f jupyterlab pandas nltk mlflow tensorflow-gpu
else
    echo "Failed to setup environment."
    exit 1
fi

conda activate ./venv
conda config --set env_prompt '({name}) '
conda env config vars set PYTHONPATH=$PYTHONPATH

pip install kaggle
kaggle competitions download -c $COMPETITION
unzip $COMPETITION.zip -d data/$COMPETITION
rm $COMPETITION.zip

conda deactivate