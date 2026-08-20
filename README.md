# CFTR Modulator Predictor

An end-to-end research pipeline for prioritizing small molecules against the
cystic fibrosis transmembrane conductance regulator (CFTR). The project combines
ChEMBL bioactivity data, RDKit descriptors and Morgan fingerprints, activity
classification, potency regression, CFTR pocket docking, developability
filters, ranked leads, structure visualization, and a local molecule-design web
application.

> **Research use only.** Predictions and docking poses are hypotheses for
> experimental prioritization. They are not clinical guidance or proof of CFTR
> modulation, mutation-specific response, safety, efficacy, or drug approval.

## Quick start

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd cftr-modulator-design
./install.sh
./run_all.sh
```

Open `results/report/index.html` for the generated analysis report. To start
the interactive molecule designer after the models have been generated:

```bash
./run_webapp.sh
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). Initial startup can
take about one minute while the models and training fingerprints are loaded.
Tool is accessible at https://cftr-modulator-design.onrender.com/

## What the project does

- Retrieves CFTR sequence, structures, ChEMBL activities, and assay context.
- Calculates 14 physicochemical descriptors and 1,024-bit Morgan/ECFP4
  fingerprints.
- Compares eight regression algorithms for compound potency.
- Saves an RBF-SVR model and paired activity-classification/potency models.
- Prepares CFTR receptors and ligands and runs AutoDock Vina across five sites.
- Ranks compounds using activity probability, potency, docking support, and
  drug-likeness.
- Exports target-matched docked complexes and PyMOL PNG visualizations.
- Serves an interactive browser application for drawing a molecule and
  obtaining instant CFTR QSAR predictions.

## Scientific scope

The principal target is direct human CFTR (`UniProt P13569`, ChEMBL target
`CHEMBL4051`). `CHEMBL5523680` is retained as a reference document, not used as
the target identifier. Approved CFTR modulators are reference controls; the
primary screening goal is prioritization of additional small molecules.

The docking branch evaluates:

- Potentiator site near TM4/TM5/TM8
- Elexacaftor-associated TMD/lasso site
- Type-I corrector site in TMD1
- ABP1 noncanonical/degenerate ATP pocket
- ABP2 canonical/hydrolytic ATP pocket

Docking scores from different pockets should not be treated as directly
equivalent experimental binding free energies.

## Repository contents

```text
cftr-modulator-design/
├── app.py                         Local molecule-design/QSAR server
├── run_webapp.sh                  Web-app launcher
├── fullpipeline.sh                Complete one-click analysis
├── run_all.sh / RUN_ALL.command   Convenience launchers
├── screen_library.sh              External-library screening launcher
├── install.sh                     Environment + Vina installer
├── requirements.txt               pip dependencies
├── environment.yml                Conda alternative
├── config/config.yaml             CFTR targets, structures, and pockets
├── src/                            Numbered analysis scripts
├── data/                           Generated/downloaded data (Git-ignored)
├── models/                         Generated joblib models (Git-ignored)
└── results/                        Generated tables/plots/report (Git-ignored)
```

## Installation

Python 3.10 or 3.11 is recommended. On macOS or Linux:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd cftr-modulator-design
./install.sh
source .venv/bin/activate
```

`install.sh` creates `.venv`, installs `requirements.txt`, and downloads a
platform-specific AutoDock Vina 1.2.7 executable into `bin/vina`. If your
platform is not supported by the installer, download Vina from its official
release page and place the executable at `bin/vina`.

Conda alternative:

```bash
conda env create -f environment.yml
conda activate cftr-modulator-design
```

## Run the complete pipeline

```bash
./run_all.sh
```

Equivalent explicit command:

```bash
./fullpipeline.sh
```

The run performs data retrieval, feature generation, mutation-response branch,
ChEMBL QSAR, receptor/ligand preparation, docking, integrated classification
and regression, ranking, plotting, and report generation. The report is written
to:

```text
results/report/index.html
```

For a faster ligand-only model build that skips docking:

```bash
./fullpipeline.sh --skip-docking
```

## Interactive molecule-design web app

The app requires the trained model files in `models/`. Generate them first with
the full pipeline or the `--skip-docking` command above, then launch:

```bash
./run_webapp.sh
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).
Open https://cftr-modulator-design.onrender.com/

The app supports molecule drawing through JSME and accepts direct SMILES,
ChEMBL molecule IDs (for example `CHEMBL25`), PubChem CIDs (for example
`CID 2244`), or chemical names (for example `aspirin`). Identifier and name
lookup requires internet access through the official ChEMBL or PubChem API;
direct SMILES prediction works locally. The app returns:

- CFTR `ACTIVE`/`INACTIVE` classification at probability threshold 0.50
- Activity probability
- RBF-SVR predicted pX and approximate nM potency
- Integrated-model predicted pX and a consensus pX
- Molecular weight, cLogP, TPSA, H-bond counts, rotatable bonds, QED, and
  Lipinski violations
- Nearest-training-compound Tanimoto similarity and an applicability-domain
  label

The JSME drawing widget is loaded from its public web distribution and requires
internet access. SMILES entry and server-side RDKit prediction remain the core
calculation path.

Instant QSAR predictions do not automatically run docking. After prediction,
the user can explicitly start a background Vina job against all five configured
CFTR pockets. The app reports the five affinities, identifies the best pocket,
renders the receptor–ligand complex, and provides the best-pose PDBQT and
complex PDB for download. Receptors and pocket geometry must first be prepared
by pipeline steps 11–12.

The molecule designer accepts a molecular structure or SMILES and therefore
runs the compound-level QSAR models. A molecule alone does not identify a CFTR
genotype and cannot be used to infer mutagenicity. Mutation–product prediction
is therefore kept as a separate command-line workflow.

If port `8765` is already occupied, either use the application already running
at that address or launch on another port:

```bash
./run_webapp.sh --port 8766
```

Shell launchers must be run with `bash`/`./`; do not run them with Python.

## Screen a new compound library

Accepted CSV columns are `compound_id,canonical_smiles`. A two-column
`SMILES ID` `.smi` file is also accepted.

```bash
./screen_library.sh path/to/compounds.csv 100
```

Example input:

```csv
compound_id,canonical_smiles
candidate_001,CCOc1ccc(NC(=O)c2ccccc2)cc1
candidate_002,CN1CCC(c2ccc(F)cc2)CC1
```

Outputs include validated features, rejected structures, candidate PDBQT files,
optional docking scores, and `results/virtual_screening_hits.csv`.

## Models

### RBF-SVR potency model

The model comparison evaluates linear regression, ridge, lasso, elastic net,
KNN, random forest, gradient boosting, and RBF-SVR. The current RBF-SVR uses:

```text
median imputation → standard scaling → SVR(kernel="rbf", C=2, epsilon=0.1)
```

Its input is 14 RDKit descriptors plus 1,024 Morgan fingerprint bits. The
training script saves the winning pipeline as a joblib bundle containing the
preprocessing pipeline and exact feature-column order.

### Integrated activity and potency models

The paired integrated models use scaffold-grouped cross-validation when enough
Bemis–Murcko scaffold groups are available. The classifier estimates active
probability and the regressor estimates pX. Docking, when available, is
supporting evidence.

### Exploratory mutation–product response model

This is a separate model whose input is a **CFTR mutation plus an approved
modulator product**. It does not accept a candidate molecule and it does not
predict whether a chemical causes genetic mutations. Score a mutation against
all configured products with:

```bash
python3 src/07_predict_modulator_response.py --mutation G551D --all-products
```

Or request one product:

```bash
python3 src/07_predict_modulator_response.py --mutation F508del --product Kaftrio
```

Configured products are Alyftrek, Kaftrio, Symkevi, Orkambi, and Kalydeco. The
current model was trained on a small hand-curated seed table and achieved
leave-one-out ROC-AUC 0.512. Its probabilities are pipeline demonstrations,
not validated mutation-specific response estimates and not treatment advice.

Never load untrusted `.joblib` files: they use Python pickle semantics and can
execute code during deserialization.

## Docking and pose visualization

The docking engine writes `results/docking_scores.csv` and pocket-specific pose
files. Post-processing scripts include:

```bash
python3 src/18_analyze_top_docking_hits.py --top 20
python3 src/19_export_top_docked_complexes.py --top 20
python3 src/20_render_best_docked_poses.py --view full --style rainbow
python3 src/21_annotate_existing_complexes.py
```

PyMOL is required only for PNG rendering. Generated poses, receptor–ligand
complexes, and PNG files are intentionally excluded from Git.

## Important validation limitations

- The RBF-SVR comparison currently uses shuffled five-fold cross-validation;
  scaffold-grouped or external validation should be added before prospective
  use.
- ChEMBL pX aggregates related potency endpoints and assay contexts. Approximate
  nM conversion is an interpretation, not necessarily a literal IC50.
- The mutation-response branch is exploratory and currently performs near
  chance. It must not be used to claim mutation-specific drug response.
- The compound QSAR cannot infer chemical mutagenicity or genotoxicity.
- Docking scores require pose inspection, receptor-quality review, rescoring,
  and experimental confirmation.
- Activity classification and regression are not calibrated clinical outputs.

## Reproducibility and generated artifacts

Downloaded structures, processed datasets, model binaries, docking poses,
figures, logs, and reports are excluded by `.gitignore`. Placeholder files keep
the directory structure in a fresh clone. Regenerate artifacts from source with
`./run_all.sh`.

Before publishing changes, run:

```bash
python3 scripts/github_preflight.py
for script in *.sh; do bash -n "$script"; done
```

If you intentionally want to distribute model binaries or large example
results, use a versioned release archive or Git LFS and document the model/data
provenance and license.

## Data and software sources

- UniProt: CFTR sequence and annotation
- RCSB PDB: experimental structures
- AlphaFold DB: predicted CFTR structure
- ChEMBL: assay and compound bioactivity records
- PubChem: reference ligand information
- RDKit: molecular descriptors, fingerprints, and depictions
- AutoDock Vina and Meeko: docking and PDBQT preparation

Respect each upstream source's terms, attribution requirements, and data
license when redistributing derived datasets.

## Contributing

Open an issue describing the scientific question, data provenance, expected
behavior, and a minimal reproducible example. Keep generated data and model
binaries out of pull requests unless explicitly required.

## License

No open-source license has been selected in this repository. Add a license
before public distribution if you want others to have explicit permission to
reuse, modify, or redistribute the code.
