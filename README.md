# OAT15A: POD, DMD and Koopman Dictionary Learning

A computational study of pressure snapshots from an OAT15A transonic buffet dataset. The project compares POD and projected DMD reconstruction, then studies polynomial EDMD, neural dictionary learning and kernel dictionary learning on five POD coordinates.

The repository contains experiment code and notebooks. Original input data, generated figures, model checkpoints and numerical results are kept outside version control. The implementations adapt established methods to this dataset; they do not establish a new identification method or validation on hardware.

## Experiments

| Notebook | Experiment | Prerequisite |
| --- | --- | --- |
| [01: Full-snapshot POD](notebooks/01_oat15a_full_pod.ipynb) | Modal energy, pressure modes and reconstruction | Original data |
| [02: POD-DMD comparison](notebooks/02_oat15a_pod_dmd_comparison.ipynb) | Stride-2 sampling, POD and projected DMD | Original data |
| [03: Classical EDMD](notebooks/03_oat15a_classic_edmd.ipynb) | Polynomial dictionaries on five POD coefficients | Original data and output of 02 |
| [04: Neural dictionary EDMD](notebooks/04_oat15a_edmd_dl_q_li.ipynb) | Alternating operator and neural dictionary updates | Original data and output of 02 |
| [05: Kernel dictionary EDMD](notebooks/05_oat15a_kedmd_dl_dual.ipynb) | A learned weighted kernel mixture | Original data and output of 02 |

`src/` contains reusable implementations, including `edmd.py`, which is required by the OAT15A experiments. `scripts/run_*.py` provides command-line entry points for the same analyses. Notebook 01 is independent; run 02, 03, 04 and 05 in that order for the complete comparison. Experiment 05 includes comparison rows from 03 and 04 only when their summary files already exist.

## Environment

Use Python 3.11 for a new environment. From the repository root:

```sh
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

The core versions are pinned to the local verification environment. Training runs on CPU; a CUDA installation is not required by this code. The notebook interface dependencies have bounded versions rather than a complete environment lock.

## Prepare the data

The repository alone is insufficient to reproduce the experiments: you need the original four input files. A verified public download URL and the dataset's complete provenance have not been established in this repository; obtain the data from its original provider. No data download or CFD simulation is performed by the scripts.

The loader expects whitespace-separated numeric text without header rows:

| File | Content | Shape in the locally verified dataset |
| --- | --- | --- |
| `coordinate.dat` | Spatial coordinates; columns are x and y | 8100 x 2 |
| `node.dat` | Four node indices per quadrilateral cell, indexed from 1 | 11260 x 4 |
| `Snapshotp.dat` | Pressure; spatial nodes in rows, time snapshots in columns | 8100 x 2000 |
| `OUTPUT.dat` | Time and lift coefficient in the first two columns | 10834 x 2 |

Place the files in this default layout:

```text
koopman-sciml/
  data/
    oat15a/
      coordinate.dat
      node.dat
      Snapshotp.dat
      OUTPUT.dat
```

Alternatively, keep the data outside the repository and set the directory once for the current terminal session. Replace the example path with the folder containing all four files.

```powershell
# Windows PowerShell
$env:OAT15A_DATA_DIR = "D:\datasets\oat15a"
```

```sh
# macOS/Linux
export OAT15A_DATA_DIR="/path/to/oat15a"
```

Data selection precedence is: explicit `--data-dir` (or configuration `data_dir`), then `OAT15A_DATA_DIR`, then the configuration's `data_root` (default: this checkout's `data/oat15a`). The original nested assignment-folder layout is also accepted through `--data-root`. Unset `OAT15A_DATA_DIR` before using `--data-root` if you want that root to take effect.

The default snapshot interval is 0.03647971526 and the snapshot start time is 110.28. The stride-2 experiments use an interval of 0.07295943052. These settings belong to the original dataset; do not reuse them for a differently sampled dataset without updating the configuration.

## Run

Run commands from the repository root, after preparing the data:

```sh
# Independent full-snapshot POD analysis.
python scripts/run_oat15a_pod.py

# Required preparation for all three EDMD experiments.
python scripts/run_oat15a_pod_dmd.py

# Run in this order to populate the final cross-method comparison.
python scripts/run_oat15a_classic_edmd.py
python scripts/run_oat15a_edmd_dl.py
python scripts/run_oat15a_kedmd_dl.py
```

Every entry point accepts `--data-dir`, `--figures-dir` and `--results-dir`; use `--help` for its options. The EDMD entry points also accept `--pod-dmd-results` if the intermediate file is stored elsewhere. For example:

```sh
python scripts/run_oat15a_pod_dmd.py --data-dir "/path/to/oat15a"
```

Experiment 02 generates `results/oat15a_pod_dmd_stride2/oat15a_stride2_pod_dmd_results.npz`. A fresh checkout will raise `FileNotFoundError` if 03, 04 or 05 is run before that file is generated. Changing the output directory of 02 requires passing the corresponding `--pod-dmd-results` to the later scripts.

The neural experiment evaluates up to 4000 training epochs and the kernel experiment up to 200, then retrains for the epoch count selected on the validation interval. Those are full experiment settings, not a fast demonstration.

To run the notebooks, start JupyterLab from the same terminal so it inherits the data setting:

```sh
python -m jupyterlab
```

Select the environment's Python 3 kernel and run the cells from top to bottom. The notebooks use the repository's default result directories. For a machine without a display, set `MPLBACKEND=Agg` in the terminal before running the scripts.

Outputs are written to `figures/<experiment>/` and `results/<experiment>/`. These directories and the original data directory are ignored by Git. Nothing in the workflow requires uploading the original dataset or generated results.

## Evaluation boundaries

The POD-DMD experiment reconstructs the sampled pressure sequence. The EDMD experiments use a contiguous 600/100/300 train/validation/test split of the 1000 stride-2 snapshots under the default settings; state normalization uses the training interval.

However, the pressure mean and POD basis are computed on the full sampled sequence before that split. The test interval therefore contributes to the representation. These are temporal prediction experiments in a fixed full-sequence POD representation, not a strictly held-out evaluation of the entire preprocessing and identification pipeline. A stricter study would fit the pressure mean and POD basis on training data and project later snapshots into that basis.

State prediction error, full pressure-fluctuation error and raw-pressure reconstruction error have different denominators and should not be compared as the same metric. For kernel EDMD, recursive state prediction with re-lifting is also distinct from a rollout using only powers of the lifted linear operator; the implementation reports both. No claim of unseen-flow-condition generalization, physical stability or control performance is made.

## Verification and method references

On 2026-11-22, all five published notebooks' code cells were rerun sequentially on the original local data, including figure generation and image references. The environment used Python 3.9.25, NumPy 2.0.2, Matplotlib 3.9.2 and PyTorch 2.5.1+cu121, with training on CPU. This verifies the existing-data workflow; it is not a clean-environment installation test or a claim of exact paper replication.

The code follows the POD-DMD transonic buffet comparison attributed to Kou et al. (2016), Li et al. (2017), *Extended dynamic mode decomposition with dictionary learning: a data-driven adaptive spectral decomposition of the Koopman operator*, and Bolager et al. (2026), *Dictionary Learning for Kernel EDMD*, Algorithm 2. Implementation choices and experiment settings should be considered when comparing with those works.
