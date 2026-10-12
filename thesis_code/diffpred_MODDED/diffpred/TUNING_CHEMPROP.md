# Tune DiffPred's Chemprop on the JKU data set

The reference is the existing `exp_configs/tox21/chemprop.yaml`, **not JKU's
published Chemprop model configuration**. The model and trainer implementations
match `diffpred_original`. Molecular features, loss, prediction processing and
logit rounding are unchanged. The JKU Hugging Face data supplies the training and validation sets.

```python
# Preview without training:
!uv run python -u tune_tox21.py --dry-run --data-dir data/tox21_jku

# Train all manually specified candidates:
!uv run python -u tune_tox21.py --device cuda --seed 0 \
  --data-dir data/tox21_jku \
  --output-root /content/drive/MyDrive/AI_thesis/diffpred_runs/diffpred_manual_v1
```

There are **29 configurations**, including the reference, in
`exp_configs/tox21/chemprop_manual_candidates.yaml`. 
Each candidate trains from scratch and changes one parameter relative to the reference.

It is possible to edit this YAML to add more combined settings; 
One-at-a-time improvements do not prove their combination is best. 
The winner is the best **tested** configuration.

| Effective parameter | Reference | Alternatives |
| --- | --- | --- |
| Message depth | 3 | 2, 4, 5 |
| Message width (`feature_hiddens`) | [300] | [128], [256], [512] |
| Message-passing dropout | 0 | 0.1, 0.2, 0.3 |
| Readout depth (`num_hiddens`) | 3 | 1, 2, 4 |
| Readout type | residual | linear |
| Activation | ReLU | SiLU, tanh |
| Aggregation | normalized sum | mean, sum |
| Message passing | bond | atom |
| Learning rate | 0.001 | 0.0001, 0.0005, 0.003 |
| Batch size | 1024 | 256, 512 |
| Weight decay | 0 | 0.00001, 0.0001 |
| Plateau reduction factor | 0.5 | 0.1 |
| Plateau patience (validation checks) | 20 | 10 |
| Plateau scheduler | enabled | disabled |

This is different than the JKU Chemprop and the original Chemprop developed
by Heid et al.
The original readout adjusts its widths by factors of two. Its initial width is
tied to message width: the reference uses 300, 150, 75. There is no separate
FFN-width control in this implementation.
`hidden_dim`, `pooling`, and `batch_norm` do not control this Chemprop baseline.
Only one value is allowed in `feature_hiddens`, because only the last entry is
used. Neither new layers nor a new optimizer or schedule are added.

Fixed by default are: 
Adam optimizer with its original defaults, baseline scheduler 
minimum LR 1e-4, maximum 800 epochs, validation every 10 epochs, early-stop patience 15
validation checks (150 epochs without AP (PR-AUC) improvement), and seed zero. 
Checkpoint selection and candidate ranking both use validation AP (PR-AUC). 
The plateau scheduler continues to monitor validation loss, same as in DiffPred.

Use `--only reference lr_0005 dropout_01` to run a subset. Replace `--seed 0`
with e.g. `--seeds 0 42 21` to rank means over multiple seeds.
No candidate is eligible until all its requested seeds complete. Optional
`--checkpoint-metric` and `--selection-metric` accept `pr_auc`, `roc_auc`, or
`loss`. Choose them before inspecting the results.

Tuning never loads or evaluates the test set. Logs stream to the notebook and
each run's `console.log`. Each fit records its effective config, checkpoint,
validation AP/ROC-AUC/loss, model size and elapsed time. Sweep outputs are
`validation_scores.csv`, `ranking.json`, `best_params.json`, `best_params.yaml`
and `best_params_diff.yaml`. Selection remains provisional until all candidates
finish. Exact ties use candidate order.

The diffusion export copies the selected molecular encoder settings, batch size,
LR, weight decay and scheduler controls. It keeps the conditioned
prediction layer, target standardization, loss weights, EMA and sampling configuration.
Baseline readout parameters are not equivalent to the conditioned layer's
parameters.

After tuning, use the exported YAMLs in 5 fresh paired-seed runs:

```python
from pathlib import Path
import subprocess

sweep = Path('/content/drive/MyDrive/AI_thesis/diffpred_runs/diffpred_manual_v1')
for config in ['best_params.yaml', 'best_params_diff.yaml']:
    for seed in [0, 42, 21, 17, 9]:
        subprocess.run([
            'uv', 'run', 'python', '-u', 'run.py',
            '--config-path', str(sweep), '--config-name', config,
            f'experiment.seed={seed}',
        ], check=True)
```

Final evaluation requires the hold-out CSV which is not provided here.
For the complete reproduction the user must have access to the test set used by JKU.
