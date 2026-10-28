# Bank Customer Churn Analysis

Exploratory data analysis of bank customer churn: **who leaves, why they leave, and what the bank can do about it.**

Mini project – Department of Artificial Intelligence and Data Science, SRM TRP Engineering College.

**Team:** Venus T (814725243183) · Sakthi M (814725243151)
**Guide:** V. Ezhil Swanly

---

## Project objectives

1. **Identify churn** – measure how many customers leave versus stay.
2. **Find patterns** – compare churn across age groups, countries, gender, activity level, balance and number of products.
3. **Recommend action** – turn the findings into practical retention ideas.

## Dataset

- Based on the public *Bank Customer Churn / Churn Modelling* dataset structure: **10,000 customers, 10 attributes**.
- Target: `exited` (1 = left the bank, 0 = stayed).
- Attributes: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

> **Note:** `messy_churn.csv` is a synthetic dataset generated to mimic the real one and to contain realistic data-quality problems. Its rates are close to, but not identical with, the public dataset. You can run the same notebook on the original Kaggle CSV (see *Using the real dataset* below).

## Repository contents

| File | Description |
|---|---|
| `Bank_Churn_AllInOne.ipynb` | Single-cell Google Colab notebook: build messy data → clean → verify → analyse → charts |
| `Bank_Churn_Colab.ipynb` | Same workflow split into step-by-step cells |
| `churn_all_in_one.py` | Plain Python version of the single-cell notebook |
| `messy_churn.csv` | Raw data with quality problems (10,290 rows) |
| `cleaned_churn.csv` | Cleaned data (10,000 rows, 0 missing values, 0 duplicates) |
| `charts/` | Visualisation PNGs (see below) |
| `Bank_Customer_Churn_Analysis.pptx` | Presentation of the findings |

## Data cleaning steps

| Problem in messy data | Fix |
|---|---|
| Inconsistent column names | Standardised to lower snake_case |
| `FRANCE`, ` Germany `, `Fra.`, `M`, `female` … | Trimmed, title-cased, mapped to France / Germany / Spain and Male / Female |
| Yes/No mixed with 0/1 | Converted to 0/1 |
| Salary and balance stored as text (`$85,000.00`) | Symbols removed, converted to numbers |
| Missing values | Median (numeric), per-country median (balance), mode (categorical) |
| Impossible values (age 150, credit score 9999, negative tenure) | Set to missing, then imputed |
| Duplicate rows and repeated customer IDs | Removed (first record kept) |
| Rows with no churn label | Dropped |
| Useless columns (`RowNumber`, `Surname`) | Dropped |

New features: `age_band` (<30, 30-39, 40-49, 50-59, 60+) and `has_balance`.

## Key findings

- Overall churn is about **20.4%** (≈2,037 of 10,000 customers).
- **Age** has the strongest link to churn; customers aged **40–59** churn the most.
- **Inactive members** churn far more than active ones.
- **Germany** has roughly double the churn of France and Spain.
- **Female** customers churn more than male customers.
- Customers with **3–4 products** show very high churn (few customers hold that many); **2 products** is the safest.
- Credit score, tenure and salary show almost no relationship with churn.

## Visualisations (`charts/`)

1. `1_churn_split.png` – retained vs churned
2. `2_correlation_heatmap.png` – correlation matrix
3. `3_churn_by_group.png` – churn by country, gender and activity
4. `4_age_and_products.png` – churn by age band and number of products
5. `5_country_activity.png` – churn by country × activity

## How to run

### Google Colab (recommended)
1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**.
2. Upload `Bank_Churn_AllInOne.ipynb`.
3. Click **Runtime → Run all**. It creates `messy_churn.csv` and `cleaned_churn.csv` and draws all charts.

### Locally
```bash
pip install pandas numpy scipy matplotlib seaborn
python churn_all_in_one.py
```

### Using the real dataset
Download `Churn_Modelling.csv` from Kaggle, upload it in the optional step 1b of `Bank_Churn_Colab.ipynb` (uncomment the upload lines), save it as `messy_churn.csv`, then run the cleaning cells.

## Recommended retention actions

1. **Re-engage inactive members** – trigger offers and outreach when activity drops.
2. **Target the 40–59 segment** – tailored advice, loyalty rewards and relationship-manager contact.
3. **Investigate Germany** – review pricing and service gaps behind the higher churn.
4. **Review 3–4 product bundles** – find out why heavy cross-selling ends in exits.

## Tools

Python · pandas · NumPy · SciPy · Matplotlib · Seaborn · Google Colab
