# Verifying the Right to Be Forgotten in Deployed Chest X-Ray Models

Code and reproduction notebook for:

> R. R. Paida, S. R. Madhireddy, A. Kasha, D. Pansheriya, T. Hazra, and D. S K,
> "Verifying the Right to Be Forgotten in Deployed Chest X-Ray Models:
> Confounded Metrics, Silent Rare-Disease Degradation, and a Calibrated Audit
> Protocol," submitted to *IEEE Transactions on Dependable and Secure
> Computing*.
> Department of Artificial Intelligence, Sardar Vallabhbhai National Institute
> of Technology, Surat, India.

---

## What this is

An audit of the evidence that approximate machine-unlearning methods report,
rather than a new unlearning method. We train retrain-from-scratch oracles —
the references most studies omit on cost grounds — and use them to calibrate
the two verification signals in standard use. Four results follow:

1. **Both standard signals are confounded by forget-set composition.** The
   forget-versus-retain loss gap is −0.0665 *before any unlearning*, and a
   loss-threshold membership attack scores 0.7517 against an oracle that
   provably never saw the data. Read against a nominal 0.5, that attack is
   reporting mostly composition.

2. **At single-patient scale exact deletion is nearly free**, and the untouched
   model already sits 0.0005 from the oracle's forget-set loss. Four published
   methods land near the oracle — and so does doing nothing.

3. **Maximising forget-set loss provably over-corrects** (Prop. 3), costing
   22–27 AUC points and creating a membership signal rather than removing one.

4. **Collateral damage concentrates in rare pathologies** (ρ = +0.85,
   p = 10⁻⁴). A scheme whose macro-average AUC falls 0.043 loses 0.140 on
   Pneumonia — invisible to headline monitoring.

We also correct the privacy accounting used in this method family: aggregate
gradient clipping inflates the L2 sensitivity by the batch size, and advanced
composition over-prices the repair. Under per-sample clipping with zCDP
composition, ε_tot = 0.59 over fifteen requests is reachable at *higher* macro
AUC than the σ = 10⁻⁴ in common use.

---

## Files

| File | What it does |
|---|---|
| `MUL_MASTER_Ultimate_final.ipynb` | Every checkpoint, oracle, attack and per-request result in the paper. Cells are labelled `S*` (setup), `B*` (baseline), `R*` (published-method replications), `C*`/`L*` (CheXpert and large-forget-set phases), `F*` (final additions), `X*` (verification and figure export). |
| `privacy_accounting.py` | Regenerates Table I and Table VII from `(η, C, B, E, σ, δ, k)` alone. Classical Gaussian, analytic Gaussian (Balle–Wang), advanced composition and zCDP (Bun–Steinke), side by side. No GPU, no data, runs in two seconds. |

If you only want to check the differential-privacy claims, run
`privacy_accounting.py`. It needs nothing but NumPy and SciPy and reproduces
every ε in the paper.

---

## Data — you must obtain this yourself

**No image data, labels, or trained checkpoints are redistributed here.**

- **NIH ChestX-ray14** — 112,120 frontal radiographs, 30,805 patients.
  Available from the NIH Clinical Center:
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- **CheXpert** — Stanford ML Group, released under a **Research Use
  Agreement** which prohibits redistribution:
  https://stanfordmlgroup.github.io/competitions/chexpert/

You need to accept the CheXpert RUA and download it directly. Trained
checkpoints are excluded from this repository for the same reason: a model
trained on CheXpert is arguably derived data under the agreement.

Point `CFG["DATA_ROOT"]` at your local copy before running.

---

## Reproducing

Requires a CUDA GPU, PyTorch, torchvision, scikit-learn, pandas, matplotlib,
scipy, tqdm.

```bash
pip install torch torchvision scikit-learn pandas matplotlib scipy tqdm
```

Run cells in label order. `S1`–`S5` set up the dataset and model; `S3` walks
the image index and takes about four minutes. `B1`–`B15` train the baseline,
build the patient-wise forget set, and run the constructed schemes.

**The patient-wise split matters.** The notebook's default `S3` split is
image-level, which leaks patients between partitions and inflates macro AUC to
0.8391 against 0.8342 patient-wise. Cells `F5`/`F6` swap in the patient-wise
partition; every headline number in the paper is patient-wise. Two studies —
the σ sweep and the sequential-deletion study — are reported on the image-level
checkpoint with its own 57-image forget set, and are labelled as such in the
paper (Section V). Confusing the two is the easiest mistake to make here.

Total compute for the full pipeline is roughly 40 GPU-hours on a single
consumer GPU, of which about 27 hours are the four retrain oracles. Oracle
training is batch-level resumable.

All runs use a fixed seed (42) for Python, NumPy, PyTorch and cuDNN.

---

## The audit protocol

If you only want the deployable part, Section VII of the paper states it as
seven checks. Once a one-off oracle exists (6.5 GPU-hours), each erasure
request costs under a minute of GPU time:

- **P0/P1** Train a retain-only oracle. Record two constants: the forget-set
  loss it reproduces, and the empirical attack floor. These replace the wrong
  nominal targets of maximal forget loss and attack AUC 0.5.
- **P2** Report ΔL_f and ΔL_r on identical samples. Never a
  forget-versus-retain comparison.
- **P3** Judge by proximity to the oracle, not magnitude. A large ΔL_f is a red
  flag, not a success.
- **P4** Run the oracle-calibrated attack. The raw threshold attack separates a
  maximally-memorising model from the oracle by 0.0297; the calibrated version
  by 0.1780.
- **P5** Gate on per-class AUC with a tighter margin below a declared
  prevalence threshold.
- **P6** Compose the privacy budget cumulatively, with zCDP rather than
  advanced composition.
- **P7** Fix the clipping before reporting any ε. Per-sample clipping is a
  one-line change and a precondition for the guarantee.

---

## Citation

```bibtex
@article{paida2026forgotten,
  author  = {Paida, Raja Rathan and Madhireddy, Samhitha Reddy and
             Kasha, Ashwith and Pansheriya, Diya and Hazra, Tanmoy and
             S K, Deeraj},
  title   = {Verifying the Right to Be Forgotten in Deployed Chest X-Ray
             Models: Confounded Metrics, Silent Rare-Disease Degradation,
             and a Calibrated Audit Protocol},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  year    = {2026},
  note    = {Under review}
}
```

---

## Acknowledgment

Computational resources were provided by the Department of Artificial
Intelligence, Sardar Vallabhbhai National Institute of Technology, Surat,
India. We thank the NIH Clinical Center for making ChestX-ray14 publicly
available and the Stanford Machine Learning Group for CheXpert.