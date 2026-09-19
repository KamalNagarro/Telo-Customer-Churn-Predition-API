# 📱 Telco Customer Churn Prediction & REST API Service

An end-to-end machine learning project for predicting telecommunications customer churn using the **IBM Telco Customer Churn Dataset**. The solution includes comprehensive data preprocessing, exploratory data analysis (EDA), custom feature engineering, Decision Tree classification models, model interpretation, and a production FastAPI REST API service.

---

## 📁 Repository Structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv                 # Primary dataset (7,043 rows)
│   └── TelcoCustomerChurn - Data Dictionary.csv # Data dictionary & field specifications
├── notebook/
│   └── churn_analysis.ipynb                   # Complete interactive Jupyter Notebook
├── model/
│   └── churn_model.pkl                        # Serialized scikit-learn Pipeline
├── app.py                                     # FastAPI REST API application
├── requirements.txt                           # Project Python dependencies
├── sample_request.json                        # Sample JSON API request payload
└── README.md                                  # Setup & execution documentation
```

---

## 📊 Executive Summary & Key Results

- **Business Objective**: Identify high-risk churn customers to enable proactive customer retention campaigns.
- **Dataset**: IBM Telco Customer Churn dataset (7,043 customer records, 21 initial attributes).
- **Target Variable**: `Churn` (Yes / No). Target class balance: 73.5% No, 26.5% Yes.
- **Engineered Features**:
  1. `TotalServices`: Count of subscribed value-added services.
  2. `TenureGroup`: Binned tenure lifecycle ranges (`0-12m`, `13-24m`, `25-48m`, `49-72m`).
  3. `AutomaticPayment`: Binary indicator for automatic bank/credit payment methods.

### 📈 Model Evaluation Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Decision Tree** | 73.73% | 50.53% | 51.34% | 50.93% | 0.6662 |
| **Regularized Decision Tree (max_depth=5)** *(Selected)* | 71.60% | 47.87% | **78.25%** | 59.40% | 0.8260 |
| **Tuned Decision Tree (GridSearch)** | 72.46% | 48.87% | **80.93%** | 60.94% | 0.8269 |
| **Logistic Regression** | 80.03% | 65.83% | 51.52% | 57.80% | 0.8447 |
| **Random Forest Classifier** | 77.43% | 56.62% | 63.99% | 60.08% | 0.8213 |

> **Business Strategic Metric Alignment**: **Recall** is prioritized over Precision because the cost of losing a customer lifetime value ($800–$2,000+) vastly exceeds the minimal cost of sending a proactive retention discount ($10–$30) to a loyal customer.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Environment Setup

Install dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Running the Jupyter Notebook

Open and run `notebook/churn_analysis.ipynb`:

```bash
jupyter lab notebook/churn_analysis.ipynb
# OR
jupyter notebook notebook/churn_analysis.ipynb
```

### 3. Launching the REST API Service

Start the FastAPI app using `uvicorn`:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Interactive OpenAPI / Swagger documentation will be available at: `http://127.0.0.1:8000/docs`

---

## 📡 API Usage & Sample Endpoint Test

### `POST /predict`

**Request Headers**: `Content-Type: application/json`

**Sample Request Body (`sample_request.json`)**:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**cURL Command**:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

**Sample API Response**:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.77
}
```

---

## 💡 Key Business Takeaways

1. **Contract Lock-in**: Month-to-month contracts account for over 42% churn. Offering small incentives for 1-year or 2-year commitments drastically reduces churn risk.
2. **First-Year Onboarding**: Early tenure (0–12 months) has the highest churn concentration. Dedicated onboarding and tech support touchpoints in month 1–6 stabilize customer retention.
3. **Automatic Payments**: Electronic check users churn at 45%. Incentivizing automatic credit card or bank transfer payments reduces churn rates by ~30%.
