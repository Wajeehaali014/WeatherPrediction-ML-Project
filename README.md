# 🌦️ Weather Predictor — KNN vs Logistic Regression vs ANN

Machine Learning term project (CS-324, Complex Engineering Problem) — Department of Computer & Information Systems Engineering, TE Batch 2022, Spring 2025.

**Group members**

| Name | Roll No. |
|---|---|
| Wajeeha Ali | CS-22055 |
| Tooba Fatima | CS-22056 |
| Asiya Fatima Tabassum | CS-22064 |

---

## 1. Project Overview

The goal is to classify a weather observation into one of **three simple classes** using six numeric weather features:

| Class | Label |
|---|---|
| Clear | 0 |
| Cloudy | 1 |
| Precipitation | 2 |

Three algorithm families were implemented, **three configurations each (9 models in total)**:

- K-Nearest Neighbors (KNN) — k = 2, 5, 15
- Logistic Regression — 3 different solver / penalty / regularization setups
- Artificial Neural Network (ANN) — 20, 50 and 80 epochs

A small interactive interface (ipywidgets) loads the saved models, takes weather values from the user and shows each model's prediction plus an **ensemble (majority vote)** result.

## 2. Dataset

- File: `weatherHistory.csv` (Kaggle — *Weather in Szeged 2006–2016*), 96,453 rows, 12 columns.
- **Input features (6):** Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees), Visibility (km), Pressure (millibars).
- **Target:** `Simple_Summary_Class`, derived from the original `Summary` column with the custom function `simplify_summary()`.

## 3. Repository Structure

```
.
├── WeatherPrediction(ML_Project).ipynb        # Full notebook: preprocessing, 9 models, interface
├── WeatherPrediction(ML_Project).pdf          # Project report
├── weatherHistory.csv               # Dataset
├── knn_model.pkl                    # Saved KNN model (k = 2)
├── logistic_regression_model.pkl    # Saved Logistic Regression (Model 1, lbfgs)
├── ann_model.h5                     # Saved ANN
├── scaler.pkl                       # Fitted StandardScaler (6 features)
├── requirements.txt
└── README.md
```

## 4. Methodology

1. **Cleaning** – dropped `Summary`, `Formatted Date`, `Loud Cover`, `Daily Summary`, `Apparent Temperature (C)`, `Precip Type`; removed duplicate rows.
2. **Label simplification** – original summaries mapped to Clear / Cloudy / Precipitation.
3. **Class balancing** – Cloudy down-sampled to 30,000; SMOTE used to raise Clear to 30,000 and Precipitation to 5,000.
4. **Outlier removal** – IQR rule (1.5 × IQR) on every numeric feature.
5. **Split** – stratified train/test split (final notebook: **70 % / 30 %**, `random_state=42`).
6. **Scaling** – `StandardScaler` fitted on the training set only, saved as `scaler.pkl`.
7. **Training & evaluation** – accuracy, classification report and confusion matrix for every model.
8. **ANN progressive training** – the ANN is trained in three stages (20, then 50, then 80 more epochs) on the same network, showing that the model becomes better trained as more epochs are added. The saved `ann_model.h5` is the final, most-trained network.

## 5. Results

### KNN
| Model | k | Train Acc. | Test Acc. |
|---|---|---|---|
| 1 | 2 | 87.96 % | 78.60 % |
| 2 | 5 | 85.76 % | 78.89 % |
| 3 | 15 | 79.93 % | 77.02 % |

### Logistic Regression
| Model | Hyperparameters | Train Acc. | Test Acc. |
|---|---|---|---|
| 1 | multinomial, lbfgs, max_iter=1000 | 61.20 % | 61.81 % |
| 2 | ovr, saga, L1, C=0.5, max_iter=2000 | 60.81 % | 61.03 % |
| 3 | multinomial, newton-cg, L2, C=1.5, max_iter=3000 | 61.21 % | 61.80 % |

### ANN (2 hidden layers × 100 units, L2 on first layer, RMSprop)
*Progressive training: the same network continues training from the previous run, to show how accuracy improves as the network is trained for more epochs. Epochs below are per run; total epochs seen are 20, 70 and 150.*
| Model | Epochs / Batch | Train Acc. | Test Acc. |
|---|---|---|---|
| 1 | 20 / 100 | 73.1 % | 72.4 % |
| 2 | 50 / 130 | 73.9 % | 73.2 % |
| 3 | 80 / 150 | 77.1 % | 75.2 % |

> Results with the **30 % test split** are given in the report (last section). ANN accuracies are slightly higher there, KNN and Logistic Regression are unchanged.

**Observations**
- KNN overfits at small k (k = 2); larger k generalises better.
- Logistic Regression underfits (~61 %) — the classes are not linearly separable with these features.
- ANN gives the best balance of accuracy and generalisation, but its validation curves are noisy.

## 6. Interface & Testing

The interface (last section of the notebook) loads the saved models and the scaler, scales the user's input with `scaler.pkl`, and shows the prediction + confidence of each model and the majority-vote ensemble.

Inputs: Temperature, Humidity, Wind Speed, Wind Bearing, Visibility, Pressure.

**Example run** (18 °C, 65 % humidity, 15 km/h, 90°, 10 km, 1005 mb):

| Model | Prediction | Confidence |
|---|---|---|
| KNN | Cloudy | 100 % |
| Logistic Regression | Cloudy | 78.2 % |
| ANN | Cloudy | 91.07 % |
| **Ensemble (majority vote)** | **Cloudy** | – |


## 7. How to Run

```bash
# 1. Clone
git clone https://github.com/Wajeehaali014/WeatherPrediction-ML-Project.git
cd WeatherPrediction-ML-Project

# 2. (Recommended) virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
jupyter notebook "WeatherPrediction(ML_Project).ipynb"
```

- Run all cells from top to bottom to retrain everything, **or** skip to the *Interface* section to use the already saved models.
- `weatherHistory.csv` must be in the same folder as the notebook.
- The interface uses `ipywidgets`, so it works in Jupyter / Colab / VS Code but **will not render on the GitHub preview**.

## 8. Saved Files

| File | Content |
|---|---|
| `scaler.pkl` | `StandardScaler` fitted on the training set (6 features, in this order: Temperature, Humidity, Wind Speed, Wind Bearing, Visibility, Pressure). Always call `.transform()` with it — never re-fit. |
| `knn_model.pkl` | `KNeighborsClassifier(n_neighbors=2)` |
| `logistic_regression_model.pkl` | `LogisticRegression` (multinomial, lbfgs) |
| `ann_model.h5` | Keras Sequential model (6 → 100 → 100 → 3, softmax) |

> The pickles were created with a specific scikit-learn version. Use the versions in `requirements.txt` to avoid `InconsistentVersionWarning`.

## 9. Known Limitations

- Only 176 real *Precipitation* samples exist; the remaining 4,800+ are synthetic.
- Validation data during ANN training is the test set.

## 10. License / Academic Note

Submitted as a coursework project for CS-324 Machine Learning. Dataset: *Weather in Szeged 2006–2016* (Kaggle).
