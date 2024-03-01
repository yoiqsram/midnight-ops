#!/bin/bash

SCRIPT_PATH=$(realpath -s "${BASH_SOURCE[-1]}")
PYTHONPATH="$(dirname "$SCRIPT_PATH")"
COMPETITION="ml-olympiad-tfugsurabaya-2024"

if [ "$1" == "--minimal" ]; then
    conda env create --prefix ./venv -f environment.minimal.yml
elif [ "$1" == "" ]; then
    conda env create --prefix ./venv -f environment.yml
else
    echo "Failed to setup environment."
    exit 1
fi

conda activate ./venv
conda config --set env_prompt '({name}) '
conda env config vars set PYTHONPATH=$PYTHONPATH

kaggle competitions download -c $COMPETITION
unzip $COMPETITION.zip -d data
rm $COMPETITION.zip

conda deactivate