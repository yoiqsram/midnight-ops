Setup for development. Make sure you have installed Anaconda in your system.
```bash
source setup.sh
```

To begin developing, run:
```bash
conda activate ./venv
mlflow server --host localhost --port 8080
```

Or if you use VS Code, you could choose set the python interpreter path to `./venv`.
