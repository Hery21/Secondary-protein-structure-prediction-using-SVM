# Protein Secondary Structure Prediction with SVM

Predicting 3-state protein secondary structure (coil / strand / helix) from primary sequence using a support vector machine. Each residue is one-hot encoded across the 20 amino acids and classified independently with an RBF-kernel SVC, with and without grid-search hyperparameter tuning.

Coursework for Bioinformatics, Universitas Gadjah Mada (2019).

## Results

Trained on RS126, evaluated on a 20% hold-out split (4,485 residues).

| Class | Residues | Precision | Recall | F1 |
|---|---|---|---|---|
| C — coil | 1971 | 0.52 | 0.69 | 0.59 |
| E — strand | 1036 | 0.39 | 0.22 | 0.28 |
| H — helix | 1478 | 0.45 | 0.40 | 0.42 |
| **Q3 (overall accuracy)** | **4485** | | | **0.48** |

Grid search over `C ∈ {0.1 … 1000}`, `gamma ∈ {1 … 0.0001}`, RBF kernel selected **C = 0.1, gamma = 1** — and produced the same 0.48 accuracy as the defaults.

## What the numbers mean

**0.48 is the majority-class baseline, not a working predictor.** Coil makes up ~47% of RS126, so a model that always answers "coil" scores about the same. The hyperparameter search confirms this: tuning didn't move the score because the bottleneck is the *features*, not the model.

The cause is the encoding. Each residue is classified from its own one-hot vector alone, with no window of neighbouring residues. Secondary structure is a property of local stretches of sequence, so a single-residue model has no signal to learn from. For context, single-sequence predictors have historically plateaued around 50–53%, while methods that supply evolutionary information (PSSM or HMM profiles) reach ~70–80%.

This repo is kept as a documented negative result: it isolates exactly how far a context-free SVM gets, and why adding sequence context is the necessary next step.

## Approach

1. **Load** `RS126.data` — alternating lines of primary sequence and DSSP-derived structure string.
2. **Filter** sequences whose primary and secondary lengths disagree.
3. **Encode** each residue with `LabelEncoder` → `OneHotEncoder`, giving a 20-dimensional input per residue. Structures map to integers: `C = 0`, `E = 1`, `H = 2`.
4. **Split** 80/20 with `train_test_split`.
5. **Train** an RBF-kernel `SVC`, first with defaults, then via `GridSearchCV`.
6. **Report** per-class precision / recall / F1 and overall Q3.

## Dataset

**RS126** — 126 non-homologous globular protein chains from Rost & Sander, sharing no more than 25% sequence identity over 80+ residues. Average length ≈ 185; composition ≈ 47% coil, 32% helix, 21% strand.

> ⚠️ **`RS126.data` is not included in this repository.** The script opens it from the working directory and will fail immediately without it. To run, place the file next to the script, with each protein as two consecutive lines:
>
> ```
> TQSHYGQCGGIGYSGPTVCASGTTCQVLNPYYSQCL
> CCCCCCEEECCCCCCCCCCCCCCCEEEEECCEEEEC
> ```

## Run

```bash
git clone https://github.com/Hery21/Secondary-protein-structure-prediction-using-SVM.git
cd Secondary-protein-structure-prediction-using-SVM
pip install "scikit-learn<1.2" numpy
python protein_secondary_structure_with_svm.py
