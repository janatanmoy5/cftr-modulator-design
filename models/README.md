# Generated models

This directory is intentionally empty in Git. Model files are generated from
the ChEMBL-derived training data and may be too large for normal Git hosting.

Create the models with:

```bash
./fullpipeline.sh --skip-docking
```

Required by `app.py`:

- `bioactivity_svr_rbf_descriptors_fp.joblib`
- `cftr_activity_classifier.joblib`
- `cftr_potency_regressor.joblib`

Do not use model files from an untrusted source: `joblib`/pickle loading can
execute arbitrary code.
