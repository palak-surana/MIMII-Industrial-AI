# CNN-LSTM Machine Fault Detection with an Agentic Active-Learning Annotation Agent (MIMII)

Does uncertainty-based active learning, run by a small multi-agent loop, reach the same cross-machine F1
with fewer labelled segments than random labelling?

```
src/config.py          all settings (every field is a --flag)          src/agents.py     the 4 agents + Coordinator
src/features.py        band-pass + log-Mel                              src/active_learning.py  LabelPool, entropy, random/uncertainty
src/preprocess.py      Phase 2  WAV -> windows                          src/analysis.py   tables, target-F1, saving, t-test, plots
src/data.py            Phase 3  machine-wise split                      src/run_experiments.py  Experiments 1-6
src/eda.py             Phase 1  required EDA                            src/run_agent_demo.py   one full agent run + logs
src/model.py/trainer.py CNN-LSTM + training                             src/xai.py        Integrated Gradients
src/baseline.py        Random-Forest comparison                         dashboard/app.py  Streamlit dashboard
tools/make_synthetic_mimii.py  fake data for smoke tests                tests/test_logic.py  logic tests (no torch needed)
```

## Step 0 - Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # install a CUDA build of torch if you have a GPU
```

## Step 1 - Get MIMII (Week 1)
Download from https://zenodo.org/records/3384388 (start with `6_dB_pump.zip`; later `6_dB_fan.zip`, `6_dB_valve.zip`,
`6_dB_slider.zip`). Unzip into `data/mimii/` so that this path exists:
`data/mimii/6_dB_pump/pump/id_00/normal/00000000.wav`  (also `abnormal/`; ids 00, 02, 04, 06).
MIMII's slide rail folder is called `slider`, so use `--machines slider`.

Optional smoke test with fake data (no download):
`python tools/make_synthetic_mimii.py --out data/synthetic` then add `--data_root data/synthetic --processed_dir data/proc_syn` to every command.

## Step 2 - Preprocess (Week 2)
```bash
python -m src.preprocess --machines pump            # builds 1 s and 2 s windows, 64 Mel, 1024 FFT, hop 512, 50 % overlap
python -m src.eda --machines pump                    # all required EDA plots/tables -> results/eda/
```
Channel 0 is used by default (`--channel -1` averages all 8). Split (fixed, Phase 3):
train = id_00, id_02 | validation = id_04 | test = unseen id_06. Change with `--train_ids ... --val_ids ... --test_ids ...`.
Use `--machines pump fan` to pool several machine types.

## Step 3 - Model, window size, supervised reference (Weeks 3-4; Experiments 1 and 2)
```bash
python -m src.run_experiments --stage exp1
```
Trains the CNN-LSTM on 100 % of the training pool for 1 s and 2 s windows, prints val/test metrics (Accuracy, Precision,
Recall/Sensitivity, Specificity, F1, AUROC, false alarms per operating hour) next to a Random-Forest baseline, and tells you the better
window. Use it for everything after: `--window_secs 1.0` (or 2.0).

## Step 4 - Random vs agentic active learning (Weeks 5-7; Experiments 3, 4)
```bash
python -m src.run_experiments --stage curves --window_secs 1.0 --seeds 0 1 2
```
For every seed, both strategies start from the SAME stratified 10 % labelled seed set, request 5 % of the pool per
round and stop at the 50 % budget; the CNN-LSTM is retrained from scratch each round. The hidden labels are released only
through `LabelPool.annotate()` (the simulated engineer). CPU too slow? add `--pool_cap 8000 --epochs 8`.

## Step 5 - Analysis (Experiments 5, 6, annotation saving)
```bash
python -m src.run_experiments --stage analysis --window_secs 1.0 --target_f1 0.80
```
Writes `results/table_f1.csv` (the 10-50 % table), `f1_curve.png` (the main graph), `summary.json` (labels needed
for the target, **annotation saving %**, paired t-tests), `exp6_final_cross_machine.csv`.
Choose the target F1 BEFORE looking at the curves (the assignment says so), e.g. from the fully-supervised F1 of Step 3.

## Step 6 - Agent demo, XAI, dashboard (Weeks 6-8)
```bash
python -m src.run_agent_demo --window_secs 1.0 --target_f1 0.80      # prints the decision block each round; saves log + model
python -m src.xai --model results/models/agent_final.pt              # Integrated Gradients on the most confident faulty test window
streamlit run dashboard/app.py -- --window_secs 1.0
```

## Design decisions to mention in the report / viva
* **Machine-wise split** prevents window leakage; test machine is never used for training or selection.
* **Labels are clip-level**: every window of an abnormal clip is labelled faulty, so some windows may be quiet. State it as a limitation.
* **Validation set**: the fully labelled validation machine is used for model selection and for the agent's stop rule (as the brief specifies); in a real plant it would also cost annotation. Say so.
* **Fairness**: same seed set, same model seeds, same retraining recipe for both strategies; only the selection rule differs.
* **Class imbalance**: positive-class weight in the loss; F1 on the faulty class is the headline metric.
* **Uncertainty = binary entropy** (max at P=0.5). Optional extension: cluster CNN-LSTM embeddings with K-means and pick uncertain samples per cluster (not included).
* Report whatever you measure: if active learning does not beat random with your data/seeds, that is a valid finding; more seeds and a lower initial fraction usually clarify it.

## Task / deliverable map
| Brief item | Where |
|---|---|
| EDA | `src/eda.py` -> `results/eda/` |
| Preprocessing, split documentation | `src/preprocess.py`, `src/data.py` (`describe_split`) |
| CNN-LSTM, metrics | `src/model.py`, `src/trainer.py`, `src/metrics.py` |
| Random baseline / uncertainty AL | `src/active_learning.py`, `Coordinator(strategy=...)` |
| Agentic framework | `src/agents.py`, `results/agent_log.json` |
| Budget table, target F1, saving | `src/analysis.py` -> `results/summary.json` |
| XAI, dashboard | `src/xai.py`, `dashboard/app.py` |
| RQ1 | Exp 1/2 (`exp1_windows.json`) | RQ2 | F1 vs % labelled | RQ3 | `table_f1.csv`, `f1_curve.png`, t-tests |
| RQ4 | labels to target (`summary.json`) | RQ5 | test-machine F1 (Exp 6) | RQ6 | agent log + dashboard |

Suggested split of work: S1 preprocess/EDA, S2 model/metrics/exp1, S3 active_learning/analysis, S4 agents/dashboard/integration.

## Dashboard in Colab / Jupyter (no Streamlit needed)
```python
import sys; sys.path.insert(0, "/content/mimii_al")
from dashboard.colab_dashboard import launch
ui = launch(results_dir=f"{R}/v2", processed_dir=P, data_root="/content/mimii_al/data/mimii", window_secs=2.0, v1_dir=R)
```
Tabs: results (curves, tables, paired tests, supervised reference), agent decisions (round slider), segment inspector
(spectrogram, waveform if raw audio is present, prediction, uncertainty, annotation status, Integrated Gradients).
The Streamlit version (`dashboard/app.py`) is kept for machines where Streamlit can run.
