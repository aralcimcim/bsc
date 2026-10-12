# Diffpred

## Installation
0. We use uv for installation. If not already installed, this needs to by installed first via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
1. Build the project environment with `uv sync`.

## Launching Experiments Directly

We use Hydra's CLI tools for dispatching experiments.
For example, to run a DiffPred run with an MLP featurization on a dataset from Molecule ACE, run

```bash
uv run python diffpred/run.py data=molecule_ace model=simple_diff_mlp \
data.dataset=CHEMBL233_Ki model.training.max_epochs=500 experiment.device=cpu
```

You can also specify a full config directly. To run DiffPred backed by CHEMPROP
on Tox21 with the splits from Polaris with the config from the paper, run

```bash
uv run python diffpred/run.py --config-path exp_configs/tox21 --config-name chemprop_diff.yaml
```
## Configuring Runs

Configuration files are found in diffpred/configs.
We have subconfigs for the model, data and general experiment components.

## Organization of the repository

High‑level layout:
- diffpred/run.py: Hydra entrypoint.
- diffpred/configs/: Hydra config tree.
  - data/: dataset configs.
  - model/: model architecture / training configs.
  - experiment/: logging, optimization, callbacks, device.
- diffpred/** (modules): core implementation (models, diffusion logic, utils).
- pyproject.toml / uv.lock: dependency specification.

## Typical workflow

1. Dry run with a short epoch count:

  ```bash
  uv run python diffpred/run.py data=<cfg> model=<cfg> model.training.max_epochs=5
  ```
2. Launch a full run; outputs land in outputs/YYYY-MM-DD/<job_name>/ created by Hydra by default.
4. Inspect outputs:
   - .hydra/config.yaml: exact resolved config (store this for reproducibility).
   - metrics (e.g. JSON / CSV, depending on logger).
   - checkpoints/ for model weights.

## Using / overriding configs

Hydra style overrides:
- Dot notation: model.training.lr=1e-4
- Add new grouped choice: data=<new_dataset_cfg>
- Multi overrides:
  ```bash
  uv run python diffpred/run.py data=molecule_ace model=simple_diff_mlp \
  model.training.batch_size=256 seed=42 experiment.device=cuda
  ```

To list available options:
```bash
uv run python diffpred/run.py --config-path diffpred/configs --config-name default --help
```

## Sweeps (multi-run)

Use Hydra multirun:
```bash
uv run python diffpred/run.py -m model.training.lr=1e-3,1e-4 experiment.seed=1,2
```
Outputs: multirun/…


