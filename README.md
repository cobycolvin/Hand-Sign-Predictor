# Hand-Sign-Predictor

End-to-end translation engine for static ASL hand signs using **two approaches**:

1. **Control (Classical ML):** SVM or Random Forest baseline.
2. **Experiment (Neural Net):** MLP in PyTorch with experiment tracking.
3. **Deployment:** Streamlit app that predicts letter + confidence from uploaded image.

Dataset: [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

---

## Project Status (All Weeks Implemented)

- ✅ **Week 1:** Classical training/evaluation pipeline + confusion matrix output.
- ✅ **Week 2:** Configurable MLP training pipeline + optional Weights & Biases logging.
- ✅ **Week 3:** Streamlit app for local inference on uploaded images.

---

## Repository Layout

```text
Hand-Sign-Predictor/
├── data/
│   ├── raw/
│   │   ├── sign_mnist_train.csv
│   │   ├── sign_mnist_test.csv
│   │   └── custom_images/           # optional extra data (A/,B/,C/...)
│   ├── interim/
│   └── processed/
├── models/
│   ├── classical/
│   └── neural/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── app/
│   ├── classical/
│   ├── data/
│   ├── neural/
│   └── utils/
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Put Kaggle data in place

Create folder and copy files:

```bash
mkdir -p data/raw
# copy sign_mnist_train.csv and sign_mnist_test.csv into data/raw/
```

---

## Week 1 — Classical Baseline (Control)

### Train SVM baseline

```bash
python -m src.classical.train_baseline \
  --train_csv data/raw/sign_mnist_train.csv \
  --test_csv data/raw/sign_mnist_test.csv \
  --model_type svm \
  --model_out models/classical/svm_baseline.joblib \
  --metrics_out reports/metrics/week1_svm_train_metrics.json
```

### Train Random Forest baseline

```bash
python -m src.classical.train_baseline \
  --train_csv data/raw/sign_mnist_train.csv \
  --test_csv data/raw/sign_mnist_test.csv \
  --model_type rf \
  --model_out models/classical/rf_baseline.joblib \
  --metrics_out reports/metrics/week1_rf_train_metrics.json
```

### Evaluate + Confusion Matrix (required deliverable)

```bash
python -m src.classical.evaluate \
  --model_path models/classical/svm_baseline.joblib \
  --test_csv data/raw/sign_mnist_test.csv \
  --confusion_out reports/figures/week1_confusion_matrix.png \
  --metrics_out reports/metrics/week1_eval_metrics.json
```

**Week 1 deliverable file:**

- `reports/figures/week1_confusion_matrix.png`

---

## Week 2 — MLP Experiments (Experiment)

Train configurable MLP:

```bash
python -m src.neural.train_mlp \
  --train_csv data/raw/sign_mnist_train.csv \
  --test_csv data/raw/sign_mnist_test.csv \
  --epochs 20 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --hidden_dims 256,128 \
  --activation relu \
  --model_out models/neural/week2_mlp.pt \
  --metrics_out reports/metrics/week2_mlp_metrics.json \
  --curve_out reports/figures/week2_loss_curves.png
```

Enable Weights & Biases logging:

```bash
python -m src.neural.train_mlp \
  --train_csv data/raw/sign_mnist_train.csv \
  --test_csv data/raw/sign_mnist_test.csv \
  --hidden_dims 512,256,128 \
  --activation gelu \
  --use_wandb \
  --wandb_project hand-sign-predictor
```

Run multiple variants for comparison (hidden layers, activation, learning rate, batch size).

**Week 2 deliverables:**

- W&B run dashboard with loss curves and validation metrics.
- Best neural checkpoint in `models/neural/`.

---

## Week 3 — Deployment (Local App)

Launch Streamlit dashboard:

```bash
streamlit run src/app/streamlit_app.py
```

App capabilities:

- Upload PNG/JPG/JPEG hand-sign image.
- Select model type (classical vs neural) and model path.
- Get predicted letter and confidence score.

Default paths expected by app:

- `models/classical/svm_baseline.joblib`
- `models/neural/week2_mlp.pt`

---

## Where to Add More Training Data

### Option A: Additional MNIST-style CSVs

1. Put extra files in `data/raw/additional/`.
2. Merge into training data in your data pipeline.
3. Keep merged outputs versioned in `data/processed/`.

### Option B: Custom image folders

Store data like:

```text
data/raw/custom_images/
├── A/
├── B/
└── ...
```

Convert to CSV:

```bash
python -m src.data.prepare_custom_images \
  --input_dir data/raw/custom_images \
  --output_csv data/processed/custom_asl_28x28.csv
```

Then combine this CSV with training data before retraining models.

---

## Useful Notes

- Keep one label-map source of truth in `src/utils/label_map.py`.
- Sign Language MNIST is static and typically excludes dynamic letters J and Z.
- Keep notebook usage optional (EDA only); training/eval/app logic stays in `src/`.

---

## Definition of Done

### Week 1

- [x] Classical model trains from CLI.
- [x] Confusion matrix generated and saved.

### Week 2

- [x] MLP training script supports architecture/activation sweeps.
- [x] Optional W&B logging supported.

### Week 3

- [x] Streamlit app runs locally.
- [x] App predicts letter + confidence on uploaded image.
