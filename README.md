# Breast Cancer Prediction (Binary Classification)

Brief project that trains a simple neural network to predict breast cancer diagnosis (benign vs malignant) using a CSV dataset.

Files
- Main script: [breast_cancer_prediction.py](c:\Users\HP\Downloads\breast_cancer_prediction.py) — key symbols: [`model`](c:\Users\HP\Downloads\breast_cancer_prediction.py), [`history`](c:\Users\HP\Downloads\breast_cancer_prediction.py), [`X_train`](c:\Users\HP\Downloads\breast_cancer_prediction.py)

Summary
- Loads `data.csv`, drops unused columns (`Unnamed: 32`, `id`) and renames `diagnosis` → `label`.
- Encodes labels with `LabelEncoder`, scales features with `MinMaxScaler`.
- Splits data with `train_test_split` (test_size=0.25, random_state=42).
- Trains a Keras Sequential model:
  - Dense(16, activation='relu') → Dropout(0.2) → Dense(1) + sigmoid
  - loss: binary_crossentropy, optimizer: adam, metrics: accuracy
  - epochs=100, batch_size=64
- Produces loss/accuracy plots and confusion matrix.

Requirements
- Python 3.8+
- numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow/keras

Install:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

How to run
1. Place the dataset file as `data.csv` in the same folder as the script.
2. Run:
```sh
python breast_cancer_prediction.py
```

Notes
- The model predicts binary labels; accuracy is computed as $accuracy = \frac{TP + TN}{TP + TN + FP + FN}$.
- The script expects the classic Wisconsin Breast Cancer CSV format (a `diagnosis` column and numeric feature columns).
- Remove or adjust hard-coded column names if your CSV differs.

Result inspection
- Training and validation loss/accuracy are plotted from the [`history`] object.
- Confusion matrix produced from test predictions.
