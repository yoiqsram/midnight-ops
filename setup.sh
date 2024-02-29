conda env create --prefix ./venv -f environment.yml
conda activate ./venv
conda config --set env_prompt '({name}) '

kaggle competitions download -c ml-olympiad-tfugsurabaya-2024
unzip ml-olympiad-tfugsurabaya-2024.zip -d data
rm ml-olympiad-tfugsurabaya-2024.zip

pip install git+https://github.com/yoiqsram/leantask.git@7f286fb

conda deactivate