# Student Academic Performance Forecasting

Deep Learning (FNN) app that forecasts a student's final grade (G3, scale 0–20)
from their first two assessment grades (G1, G2) and background information.
Built on the UCI "student-mat" dataset.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (the demo) |
| `retrain.py` | Trains the model from scratch (optional — a trained model is already included) |
| `requirements.txt` | Pinned dependencies for Streamlit Cloud |
| `model/` | Trained model + preprocessing objects + evaluation results |

## 1. Run it locally first (recommended before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. Confirm the sidebar works and a forecast
appears before you push to GitHub — it's much faster to debug locally than on
Streamlit Cloud.

## 2. Push to GitHub

```bash
cd student-performance-forecasting     # this folder
git init
git add .
git commit -m "Student performance forecasting app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

If you don't have a repo yet: go to github.com → **New repository** → give it
a name → **Create repository** (don't add a README there, you already have
one) → then run the commands above with that repo's URL.

## 3. Deploy on Streamlit Community Cloud (free)

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. Click **Create app** → **From existing repo**.
3. Pick your repository, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**.

First build takes a few minutes (installing TensorFlow). You'll get a public
URL like `https://<something>.streamlit.app` — that's what you submit / open
during the practical demo.

## Notes and known constraints

- **Free-tier RAM (≈1 GB):** TensorFlow is the heaviest dependency here. This
  app uses `tensorflow-cpu` (lighter than full `tensorflow`) and a small FNN,
  which should fit comfortably. If the app ever crashes with a memory error
  on Streamlit Cloud, that's why — the fix is trimming `requirements.txt`
  further or moving to a paid tier, not a bug in the app itself.
- **scikit-learn version:** `preprocessor.pkl` and `scaler.pkl` were saved
  with scikit-learn 1.9.1. `requirements.txt` allows 1.5–1.10 for
  compatibility; if you see an `InconsistentVersionWarning` in the logs it's
  informational, not an error — the app still works. If predictions ever look
  wrong after a scikit-learn upgrade, re-run `retrain.py` to regenerate
  `preprocessor.pkl`/`scaler.pkl`/the model together.
- **Model performance (from `metrics.pkl`):** RMSE ≈ 2.97, MAE ≈ 2.24,
  R² ≈ 0.57 on held-out test data. The app shows these on the page, and
  reports a rough ± MAE range next to each forecast, so the demo shows the
  model being evaluated, not just used — this maps directly to the "Apply
  Model Evaluation Techniques" section of the rubric.
- **Reproducibility:** `retrain.py` uses relative paths (`Path(__file__)`)
  instead of the original hardcoded `C:\Users\...` path, so anyone who clones
  the repo can regenerate the model, preprocessor, scaler and metrics with
  `python retrain.py` (needs `model/student-mat.csv`, which is included).
- **Note on `student_performance_model.keras`:** your upload included this
  second file alongside `student_performance_forecasting_model.keras`. Only
  the "forecasting" one is used by `app.py` (it matches `preprocessor.pkl`
  and `feature_information.pkl`). If the other file is a different/older
  model, it isn't part of this deployment — leave it out of the repo, or
  tell me its story if you want it included too.

## Responsible-use note (task requirement #8)

The app displays a "Responsible Use" notice after every forecast: results are
an estimate from a model trained on a limited historical dataset, meant to
support (not replace) teacher judgment, and may be less reliable for students
whose profile differs a lot from the training data (e.g. unusual absence
counts).
