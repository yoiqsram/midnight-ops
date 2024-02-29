conda env create --prefix ./venv -f environment.yml
conda activate ./venv
conda config --set env_prompt '(./venv) '

pip install git+https://github.com/yoiqsram/leantask.git@7f286fb

kaggle competitions download -c ml-olympiad-tfugsurabaya-2024
unzip ml-olympiad-tfugsurabaya-2024.zip -d data
rm ml-olympiad-tfugsurabaya-2024.zip

conda deactivate