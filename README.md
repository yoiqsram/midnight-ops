## Environment Setup
Setup for development in Linux/WSL. Make sure you have installed Anaconda in your system.
```bash
source setup.sh
```

The setup script will install full environment with `tensorflow-gpu` and `nltk`. If you want only to install the basic environment, use this:
```bash
source setup.sh --minimal
```

To begin developing, run:
```bash
source venv/bin/activate
mlflow server --host localhost --port 8080
```

Or if you use VS Code or Jupyter Notebook, you could choose set the python interpreter path or kernel from `./venv`.

If you're not using Linux/WSL, follow the command in `setup.sh` and adjust accordingly to your OS.

## How to Collaborate?

### Repo Collaboration
1. Ask me to invite you as collaborators.
2. Clone this repository on your locals.
3. Follow the environment setup guide.
4. Create a branch to start develop. If you're developing notebook, put it in `notebooks` folder. If you're developing script, put it in `scripts` folder.
5. If you're done, create a PR and notice others to review your code.

### Live Collaboration
1. Sometimes we held a live session on **VS Code**. You could join by installing **VS Code** and **Live Share** extension on your local.
2. **Live Share** host will give you the session link.
3. You could edit, run, and monitor model runs using the host machine.

### General Rules
1. Follow the template given in the `notebooks` folder.
2. Use compatible python packages with `MLFlow` (i.e. `scikit-learn`, `tensorflow`, `pytorch`, etc) and make sure to build model by following their API from preprocess to inference.
