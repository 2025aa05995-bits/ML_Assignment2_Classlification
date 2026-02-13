
# Bank Marketing Classification — End‑to‑End ML (Streamlit)

> **Machine Learning Assignment**

## a. Problem statement
Build and compare multiple classification models to predict whether a customer subscribes to a term deposit during a bank’s telemarketing campaign. The pipeline covers data preparation, model training, held‑out evaluation, and a Streamlit UI for interactive inference and metric visualization.

## b. Dataset description
We use the **UCI Bank Marketing** dataset (binary classification: target `y` ∈ {`yes`,`no`}). In line with recommended practice, the `duration` column (last contact length) is excluded to avoid data leakage. Key inputs include: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome`. Target: `y` (`yes`/`no`).

## c. Models used (with evaluation metrics)
Run `python model/train_bank_marketing.py` to train and generate metrics. Paste the values into the table below.

| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| kNN |  |  |  |  |  |  |
| Naive Bayes (Gaussian) |  |  |  |  |  |  |
| Random Forest (Ensemble) |  |  |  |  |  |  |
| XGBoost (Ensemble) |  |  |  |  |  |  |

## Observations on model performance
Add concise bullets on what you observe for each model after training (bias/variance, class imbalance effect, top features, calibration, speed, etc.).

| **ML Model Name** | **Observation about model performance** |
|---|---|
| Logistic Regression |  |
| Decision Tree |  |
| kNN |  |
| Naive Bayes (Gaussian) |  |
| Random Forest (Ensemble) |  |
| XGBoost (Ensemble) |  |

## How to reproduce
1. **Install deps**: `pip install -r requirements.txt`  
2. **Train**: `python model/train_bank_marketing.py`  
   - Saves pipelines to `model/artifacts/*.joblib` and metrics to `model/metrics_summary.csv` & `model/metrics_summary.md`.  
3. **Run Streamlit locally**: `streamlit run app.py`  
4. **Deploy** (Streamlit Cloud): New App → pick repo → branch → `app.py` → Deploy.

## BITS Virtual Lab proof
Run the training script on BITS Virtual Lab and capture a screenshot of successful execution (include it in your submission PDF).
