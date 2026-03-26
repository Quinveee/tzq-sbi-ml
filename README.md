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

# Project structure
Below is an overview of where the main pieces of the codebase are located.

```text
tzq-sbi-ml/
├── main.py                    # Main Hydra entrypoint
├── requirements.txt           # Python dependencies
├── README.md
├── conf/                      # Hydra config groups and defaults
│   ├── config.yaml            # Main default config
│   ├── hydra.yaml             # Hydra runtime/output behavior
│   ├── _auto/                 # Auto-derived config fragments
│   ├── dataset/               # Dataset config presets (e.g. 1d/3d)
│   ├── exp/                   # Experiment type configs (ratio/local/etc.)
│   ├── launcher/              # Launcher backend configs
│   ├── model/                 # Model config presets
│   └── preprocessing/         # Preprocessing config presets
├── experiments/               # Training/eval logic, losses, plotting, utils
│   ├── base/
│   ├── features/
│   ├── limits/
│   ├── particles/
│   ├── plotting/
│   ├── ensemble.py
│   ├── experiment_histos.py
│   ├── logger.py
│   ├── losses.py
│   └── utils.py
├── models/                    # Model implementations and model-specific modules
│   ├── configs.py
│   ├── mlp.py
│   ├── transformer.py
│   ├── modules/
│   ├── MIParT/
│   └── ParT/
├── wrappers/                  # Wrapper interfaces around model backends
│   ├── base_wrapper.py
│   ├── mlp_wrapper.py
│   ├── transformer_wrapper.py
│   ├── lgatr_wrapper.py
│   ├── embed.py
│   └── utils.py
├── helpers/                   # Utility helpers (config derivation, etc.)
│   └── derive_config.py
├── launchers/                 # Launcher implementations (local / HTCondor)
│   ├── local.py
│   └── htc.py
├── workers/                   # Worker-side execution utilities
│   └── worker.py
├── scripts/                   # Convenience scripts and job files
│   ├── run.sh
│   ├── run_test.sh
│   ├── test.job
│   └── test.py
├── tests/                     # Unit/integration tests
│   └── test_lorentz_equivariance.py
├── images/                    # Static assets/figures grouped by dataset
│   ├── 1d/
│   └── 3d/
├── inputs/                    # Input archives / raw generated inputs
├── outputs/                   # Hydra single-run outputs (timestamped)
├── multirun/                  # Hydra multirun outputs (timestamped)
├── run_dirs/                  # Run directories grouped by dataset
│   ├── 1d/
│   └── 3d/
├── runs/                      # Experiment tracker artifacts (e.g. wandb)
│   ├── model_experiment/
│   └── wandb/
└── logs/                      # Batch/job and experiment logs
```

Notes:
- `outputs/` and `multirun/` are Hydra-generated runtime folders and can grow quickly.
- `inputs/` and `logs/` may contain many generated/intermediate files from repeated runs.

# Usage
We use [Hydra](https://hydra.cc/) to manage different experiment configurations, so basic familiarity is recommended. If you want to use your own datasets, create appropriate files or edit existing ones in `conf/dataset`. 

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