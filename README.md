# 🚜 Heavy Equipment Price Prediction

### Machine Learning Project for Predicting Heavy Equipment Transaction Values

> An end-to-end machine learning project focused on predicting the transaction value of heavy equipment using historical asset, operational, transactional, regional, and technical specification data.

---

## 📌 Project Overview

Accurately estimating the value of used heavy equipment is influenced by several factors, including:

- Equipment age
- Operational usage
- Transaction timing
- Geographic region
- Product configuration
- Equipment specifications
- Utilization level
- Historical transaction patterns

This project develops a machine learning pipeline to predict the **`TargetValue`** of heavy equipment transactions.

The project focuses not only on model performance but also on **data exploration, feature engineering, categorical encoding, model comparison, hyperparameter optimization, and validation**.

---

## 🎯 Objective

The primary objective is:

> **Build a machine learning model capable of predicting heavy equipment transaction values with minimum Root Mean Squared Logarithmic Error (RMSLE).**

The target variable is:

```text
TargetValue
```

The evaluation metric is:

```text
RMSLE — Root Mean Squared Logarithmic Error
```

RMSLE is particularly suitable for this problem because it evaluates relative differences between predicted and actual values while reducing the influence of extremely large target values.

---

## 📊 Dataset

The dataset contains approximately **138K observations and 50 variables**, covering transaction, asset, operational, regional, and technical specification information.

### Important Variables

| Feature | Description |
|---|---|
| `TargetValue` | Transaction value to be predicted |
| `TransactionID` | Unique transaction identifier |
| `AssetID` | Identifier of the equipment/asset |
| `ProductConfigID` | Technical configuration identifier |
| `TransactionDate` | Date of the transaction |
| `RegionCode` | Geographic region |
| `ManufactureYear` | Year the equipment was manufactured |
| `OperationalHoursMeter` | Lifetime operational usage |
| `UtilizationTier` | Equipment utilization category |
| `Spec_FullDescriptor` | Detailed technical specification |
| `Spec_BaseClass` | Broad equipment specification category |
| `Spec_SubClass` | More specific equipment category |
| `FunctionalClassification` | Operational classification |

The dataset also contains numerous technical specification and configuration variables describing equipment characteristics.

---

# 🔍 Exploratory Data Analysis

A detailed EDA was performed to understand the structure and characteristics of the dataset before modeling.

### Areas investigated

- Dataset dimensions and data types
- Numerical and categorical variables
- Missing-value patterns
- Cardinality of categorical features
- Target-value distribution
- Numerical feature distributions
- Relationships between important variables and the target
- Temporal characteristics
- High-cardinality categorical variables
- Potentially unusual or extreme observations

### Key observations

The dataset contains a mixture of:

```text
Numerical Features
        +
Categorical Features
        +
High-Cardinality Identifiers
        +
Technical Specifications
        +
Transaction Information
```

This combination makes feature representation particularly important for achieving good predictive performance.

---

# 🧹 Data Preprocessing

Different preprocessing strategies were used according to feature type.

### Numerical Features

Numerical variables were processed using appropriate preprocessing techniques, including:

- Missing-value handling
- Numerical transformation where required
- Feature scaling for models where appropriate

### Categorical Features

Several encoding strategies were investigated:

- One-Hot Encoding
- Frequency Encoding
- Target Encoding
- Group/statistical target-based features

High-cardinality variables required special attention because direct one-hot encoding could produce a very large feature space.

---

# 🛠️ Feature Engineering

Feature engineering was one of the major components of this project.

### 📅 Date Features

`TransactionDate` was transformed into meaningful temporal features such as:

- Transaction year
- Transaction month
- Transaction quarter
- Other date-related components

These features allow the model to capture temporal patterns in equipment values.

### 🏗️ Asset Age

A derived feature representing the age of the equipment at the time of transaction was created using:

```text
AssetAge = TransactionYear - ManufactureYear
```

This provides a more direct representation of equipment depreciation than using manufacture year alone.

### 🔢 Frequency-Based Features

Frequency encoding was applied to selected categorical variables to represent how frequently particular configurations or categories occur.

This provides useful information while avoiding the extremely large dimensionality that can result from one-hot encoding high-cardinality features.

### 📈 Target-Based Statistical Features

Historical target statistics were investigated for selected categorical/grouping variables, including statistics such as:

- Mean
- Median
- Minimum
- Maximum
- Standard deviation

These features provide additional information about historical transaction-value patterns associated with different configurations.

> **Note:** Target-derived features require careful leakage control. In a production implementation, these features should be generated using leakage-safe out-of-fold strategies.

---

# 🤖 Machine Learning Models

Multiple tree-based models were evaluated to compare different approaches to nonlinear tabular prediction.

### Models evaluated

| Model | Purpose |
|---|---|
| 🌲 Random Forest | Strong bagging-based baseline |
| ⚡ LightGBM | Efficient gradient boosting |
| 🚀 XGBoost | Gradient boosting with strong nonlinear modeling capability |

The models were evaluated using the same validation objective so their performance could be compared fairly.

---

# 🎯 Hyperparameter Optimization

Hyperparameter optimization was performed using:

```text
RandomizedSearchCV
```

Rather than exhaustively testing every possible combination, randomized search samples a predefined number of hyperparameter configurations.

The search was used to investigate parameters controlling:

- Number of estimators
- Tree depth
- Learning rate
- Row sampling
- Feature sampling
- Minimum child weight
- Regularization
- Model complexity

This provided an efficient way to search a large hyperparameter space while keeping computational cost manageable.

---

# 📊 Model Performance

The evaluated models achieved approximately the following validation performance:

| Model | Validation RMSLE | Result |
|---|---:|---|
| Random Forest | **~0.1996** | Strong baseline |
| LightGBM | **~0.1934** | Strong boosting model |
| **XGBoost** | **~0.1921** | 🏆 Best |

### 🏆 Best Model

**XGBoost**

```text
Validation RMSLE ≈ 0.1921
```

XGBoost achieved the lowest validation RMSLE among the evaluated models and was therefore selected as the final model.

---

# 🔬 Why XGBoost?

The dataset contains nonlinear relationships and interactions between variables such as:

```text
Asset Age
      +
Operational Usage
      +
Transaction Time
      +
Region
      +
Product Configuration
      +
Technical Specifications
```

Tree-based boosting models can naturally capture nonlinear relationships and interactions between these variables.

Among the evaluated approaches, XGBoost produced the strongest validation performance, making it the preferred final model for this project.

---

# 📈 Model Development Pipeline

The overall workflow can be summarized as:

```text
                 ┌───────────────────┐
                 │   Raw Dataset     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Data Quality    │
                 │      Analysis     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │       EDA         │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Preprocessing    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Feature Engineer. │
                 └─────────┬─────────┘
                           │
                           ▼
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          Random        LightGBM      XGBoost
          Forest
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Hyperparameter    │
                 │    Optimization   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Model Comparison  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Best Model: XGB  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Test Prediction   │
                 │   & Submission    │
                 └───────────────────┘
```

---

# 🧠 Key Learnings

### 1. Feature engineering matters

Performance improvements were not obtained simply by increasing model complexity. Representing the underlying asset characteristics appropriately was an important part of the modeling process.

### 2. High-cardinality categorical variables require careful treatment

Variables such as product configurations and detailed specifications can contain a very large number of unique values.

Using only one-hot encoding can result in a very high-dimensional representation. Frequency and target-based representations provided alternative ways to represent these variables.

### 3. Tree-based boosting works well for heterogeneous tabular data

The combination of numerical, categorical, temporal, and engineered features makes gradient-boosting algorithms a strong choice for this problem.

### 4. Model performance must be interpreted carefully

A strong validation score does not automatically guarantee the same performance on future transactions.

Potential challenges include:

- Distribution shift
- Rare configurations
- Unseen categories
- Changes in market conditions
- Temporal changes
- Target-encoding leakage

---

# ⚠️ Limitations

The current evaluation has several limitations that should be considered before production deployment.

### Random validation split

The current validation strategy uses a random split. A temporal validation strategy could provide a more realistic estimate of performance on future transactions.

### Target-based features

Target-derived features can introduce leakage if the target of the same observation is used when constructing its feature representation.

A production implementation should use **out-of-fold target encoding**.

### Rare categories

High-cardinality features may contain categories with very few observations, making predictions for rare configurations less reliable.

### Distribution shift

Future transactions may differ from historical observations in terms of:

- Equipment types
- Regions
- Market conditions
- Utilization
- Product configurations

---

# 🚀 Future Improvements

Several improvements could be explored in future versions:

- [ ] Leakage-safe out-of-fold target encoding
- [ ] Temporal validation
- [ ] Repeated cross-validation
- [ ] Train/test distribution-shift analysis
- [ ] SHAP-based model interpretation
- [ ] Detailed residual/error analysis
- [ ] Subgroup performance analysis
- [ ] Prediction uncertainty estimation
- [ ] Model ensembling
- [ ] Production deployment

---

# 🧰 Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

### Hyperparameter Optimization

- Scikit-learn `RandomizedSearchCV`

---

# 📁 Project Structure

```text
heavy-equipment-price-prediction/
│
├── README.md
│
├── notebook/
│   └── 24f1002331-notebook-2026t2.ipynb
│
├── data/
│   └── README.md
│
└── submission/
    └── submission.csv
```

> Dataset files are not included in this repository when restricted by competition or data-usage rules.

---

# 📓 Notebook

The complete implementation, including:

- EDA
- Data preprocessing
- Feature engineering
- Encoding strategies
- Model training
- Hyperparameter optimization
- Model comparison
- Final prediction

is available in the project notebook.

---

# 🏁 Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting heavy equipment transaction values.

The final experiments showed that careful feature engineering and appropriate handling of high-cardinality categorical variables, combined with nonlinear boosting models, can provide strong predictive performance.

Among the evaluated models, **XGBoost achieved the best validation RMSLE of approximately 0.1921** and was selected as the final model.

The project also highlights an important practical lesson in machine learning:

> **Good model performance is not only about choosing a powerful algorithm. Reliable feature representation, validation strategy, leakage prevention, and understanding the data are equally important.**

---

## 👨‍💻 Author

**Javed Ali**

Data Science | Machine Learning | AI

---

⭐ If you find this project useful, consider giving the repository a star.
