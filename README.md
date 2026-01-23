# Get started
Clone the repository
```bash
git clone https://github.com/arturoam00/tzq-sbi-ml
```

To install dependencies in a virtual environment, do
```bash
cd tzq-sbi-ml
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

You can optionally download the paper datasets for reproduction or testing from [Google Drive](https://drive.google.com/file/d/1LLmCnUtkik1bB7CngY-CkDLROsnnIFKh/view?usp=sharing).

# Usage
We use [Hydra](https://hydra.cc/) to manage different experiment configurations, so basic familiarity is recommended. If you want to use your own datasets, create appropriate files or edit existing ones in `conf/datasets`. 

To keep the user interface cleaner, we derive many parts of the configuration based on the user input. Everything that is automatically derived is in `conf/_auto`. The final configuration object is completely specified once the fields `exp`, `model` and `dataset` are set, which can be done from the command-line. For example, to run an experiment for *likelihood ratio regression* with the *MLP* model and for the *one dimensional* dataset, run:
```bash
python main.py exp=ratio model=mlp dataset=1d
```

You can edit the default configurations just overriding them from the CL like
```bash
python main.py exp=local model=lgatr dataset=3d train.epochs=25 devices.device=cpu
```
By default, we assume GPU availability. Set the flag `devices.device=cpu` or edit `conf/config.py` as above if this is not the case.
You also want to check that `devices.eval=cpu` is set.

## Multirun
Hydra makes it easy to run several experiments with different configuration overrides. Just pass the multirun flag `-m` or `--multirun` and use commas to separate values. For example
```bash
python main.py exp=local,ratio model=lgatr,transformer dataset=3d -m
```
The above will create a 2x2 experiment grid and run each combination serially.

## HTCondor integration
We provide a [HTCondor](https://htcondor.org/) launcher to run the experiments. To use it, just specify the option `launcher=htcondor` (default is `local`). For example
```bash
python main.py exp=histo dataset launcher=htcondor
```
This is specially powerful in combination with Hydra's multirun feature: you can submit several jobs with different configuration overrides with just one command. The following would submit 36 jobs, corresponding to all (ML) experiments in the paper
```bash
python main.py exp=ratio,local model=mlp,transformer,lgatr dataset=1d,3d data.run=1,2,3 launcher=htcondor --multirun
```
The `data.run` field just modifies the output directory, handy to run multiple identical runs.