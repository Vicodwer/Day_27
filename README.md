# 💊 Pharmalink – B2B Medicine Distribution Platform
### Churn Prediction Case Study
> PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar — Week 05, Day 26

---

## 📌 Problem Statement

Pharmalink connects **1.8M+ retail pharmacies and small hospitals** across 700+ Indian cities with drug manufacturers and distributors. The Revenue Analytics team identified that **15% of pharmacies active 4 months ago have placed no orders in the last 30 days**.

> **Churn Definition:** A pharmacy is "churned" if it does not reorder within 30 days of its last purchase.

---

## 📂 Dataset Overview

| Property | Details |
|---|---|
| Timeframe | 15 months |
| Scale | 1.5 million pharmacies |
| Features | 42 |
| Target Variable | `will_churn_next_30d` (1 = Churn, 0 = Active) |

### Feature Categories

**Behavioral**
- `last_order_date`, `order_frequency_60d`, `avg_order_value`, `app_sessions_30d`

**Financial**
- `credit_utilization_ratio`, `payment_delay_days`, `credit_limit`, `outstanding_amount`

**Engagement**
- `product_diversity_score`, `return_rate`, `complaints_logged`

**Demographic**
- `city_tier`, `years_of_operation`, `store_type` (independent / chain)

---

## 🧠 Q1 — Problem Identification

| Question | Answer |
|---|---|
| Supervised or Unsupervised? | **Supervised Learning** — labeled target `will_churn_next_30d` exists |
| Classification or Regression? | **Binary Classification** — output is 0 or 1 |
| ML Category | **Customer Churn Prediction** (Propensity Modeling) |

**Justification:**
- Supervised because historical churn labels are available for training.
- Classification because the output is a discrete binary class, not a continuous value.
- Specifically a propensity model — it estimates the *probability* of churn, not just a hard label.

---

## 🔁 Q2 — 5-Stage ML Pipeline

| Stage | Pharmalink-Specific Action |
|---|---|
| **Data Collection** | Pull 15 months of transactional history — `last_order_date`, `order_frequency_60d`, `avg_order_value` — joined with financial records like `credit_utilization_ratio` and `payment_delay_days` |
| **Data Preprocessing** | Handle missing values in `app_sessions_30d`, engineer `days_since_last_order`, encode `city_tier` and `store_type` as categorical variables |
| **Model Training** | Train XGBoost/LightGBM using `order_frequency_60d`, `credit_utilization_ratio`, and `product_diversity_score` as key predictors |
| **Model Evaluation** | Evaluate using **Recall** and **AUC-ROC** — minimizing false negatives (missed churners) is the priority |
| **Deployment** | Run daily inference; flag pharmacies with churn probability above threshold; route high-value accounts (`avg_order_value` > ₹5L) to the sales team |

---

## 📊 Q3 — Evaluation Strategy

### CRO's Insight (Rohan Mehta):
> *"We can afford to call 500 pharmacies who were never going to churn. But if we lose a high-revenue hospital account placing ₹8 lakh monthly orders without warning, that's unacceptable."*

### Priority: **RECALL**

| Metric | Formula | Why It Matters |
|---|---|---|
| **Recall** | TP / (TP + FN) | Minimizes missed churners (False Negatives) |
| Precision | TP / (TP + FP) | Minimizes wasted calls (False Positives) |

**Business Trade-off:**
- A false positive = a wasted sales call → low cost
- A false negative = losing an ₹8L/month account without warning → unacceptable cost

> ✅ Optimize for **high recall**. Accept more false alarms to ensure no high-value churner goes undetected.
>
> 💡 Recommended metric: **Recall @ Top Decile** — among top 10% highest-risk pharmacies, what % actually churn?

---

## 🏗️ Q4 — Advanced Thinking: Handling Tier-3 Behavior

### The Problem
Tier-2 and Tier-3 pharmacies place large but **infrequent** bulk orders. A 45-day gap is normal for them — but a global 30-day churn window would incorrectly flag them.

### Three Options Evaluated

| Approach | Verdict | Reason |
|---|---|---|
| **One global model** | ❌ Weak | Biased toward Tier-1 behavior; high false positives for Tier-3 |
| **Separate models per city tier** | ✅ Strong | Each model learns tier-specific baseline behavior |
| **Interaction features** | ✅ Strong | Lets a single model understand tier-conditional behavior |

### Recommended Approach: **Combine Separate Models + Interaction Features**

**Engineered features to add:**
```
city_tier × order_frequency_60d         → interaction term
days_since_last_order / avg_reorder_gap → relative inactivity score
is_bulk_buyer = 1 if avg_order_value > threshold
```

**Advanced Fix:** Use a **personalized churn window** per pharmacy — instead of a fixed 30-day window, define churn as: `days_since_last_order > (2 × pharmacy's historical avg reorder gap)`.

---

## 🛠️ Tech Stack (Recommended)

```
Language     : Python 3.10+
Libraries    : pandas, scikit-learn, xgboost, lightgbm, matplotlib, seaborn
Modeling     : XGBoost / LightGBM (Binary Classification)
Evaluation   : Recall, AUC-ROC, Precision-Recall Curve
Deployment   : Batch scoring pipeline (daily)
```

---

## 📁 Repository Structure

```
pharmalink-churn/
│
├── data/
│   └── pharmacy_orders_sample.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── outputs/
│   └── churn_scores.csv
│
└── README.md
```

---

## 📎 Submission

- Submit notebook + GitHub repository link on LMS
- Ensure all cells are executed with outputs visible
- Duration: 60–75 minutes

---

*IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering*
