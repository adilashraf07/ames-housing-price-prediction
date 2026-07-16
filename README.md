# 🏠 Ames Housing Price Prediction

An end-to-end machine learning regression project predicting residential house sale
prices using the **Ames Housing dataset**, loaded directly from OpenML via
scikit-learn. Built as a complete, reproducible pipeline — from raw data to a
trained model making predictions on a custom, unseen house.

## Project Overview

House price prediction is a classic regression problem: given a set of features
describing a property (size, quality, age, location, etc.), predict its sale price.
This project walks through the full workflow a data scientist would follow on a
real tabular dataset, with an emphasis on doing each step correctly and explaining
*why*, not just *what*.

## What's Inside

- **Data exploration** — structure, data types, duplicates, missing values
- **Data preprocessing** — dtype correction, duplicate removal
- **Missing value handling** — context-aware strategies (median / mode / "None" for
  absent-feature columns), not blanket imputation
- **Exploratory Data Analysis** — distributions, boxplots, correlation heatmap,
  scatter plots, pairplot, each with a written interpretation
- **Feature engineering** — one-hot encoding, train/test split (80/20,
  `random_state=42`), feature scaling
- **Model training** — Linear Regression, Decision Tree Regressor, Random Forest
  Regressor
- **Evaluation** — MAE, MSE, RMSE, R² for every model, in a sorted comparison table
- **Model comparison** — bar charts of R² and RMSE across models
- **Diagnostics** — actual vs. predicted plot, residual plot, error distribution
- **Feature importance** — top 15 features from the Random Forest
- **Custom prediction** — a hand-built example house, run through the full
  preprocessing pipeline and priced by the best model
- **Written conclusion** — summary, key findings, and future improvements

The best-performing model is selected **programmatically** from the results table
(not hardcoded), so the diagnostics, feature importance, and custom prediction
sections automatically use whichever model wins.

## Results

Evaluated on a held-out 20% test set (292 houses):

| Model                        | MAE ($)   | MSE         | RMSE ($)  | R² Score |
|-------------------------------|-----------|-------------|-----------|----------|
| **Random Forest Regressor**   | **17,663.57** | 8.36 × 10⁸ | **28,922.10** | **0.891** |
| Decision Tree Regressor       | 28,486.87 | 1.90 × 10⁹  | 43,582.30 | 0.752    |
| Linear Regression              | 24,759.40 | 8.53 × 10⁹  | 92,377.86 | -0.113   |

**Best model: Random Forest Regressor** — explains ~89% of the variance in sale
price, with an average error of ~$17.7K on houses ranging roughly from $35K to $755K.

**A more interesting finding:** plain Linear Regression scored a **negative R²**,
meaning it performed *worse* than simply predicting the average sale price every
time. This isn't a bug — it's the expected result of feeding 272 one-hot encoded
features (many sparse/rare categories) into unregularized OLS, which is highly
sensitive to multicollinearity between those dummy columns. Ridge or Lasso
regression would very likely fix this, which is why it's listed as a follow-up
below. Calling this out explicitly, rather than hiding a weak result, is
intentional — it's a more useful signal of understanding than a uniformly "good"
table would be.

**Top predictive features** (Random Forest importance): `OverallQual` dominates at
~56% importance, followed by `GrLivArea` (~12%), `2ndFlrSF`, `TotalBsmtSF`, and
`BsmtFinSF1` — consistent with real-world housing intuition (quality and size drive
price far more than smaller amenities).

**Example prediction:** a custom hypothetical house (quality 8/10, 2,200 sq ft
living area, 3-car garage, built 2015) was priced by the Random Forest model at
**$282,186.79**.

## Tech Stack

- Python
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — modeling, preprocessing, evaluation

## Dataset

[Ames Housing dataset](https://www.openml.org/search?type=data&id=42165) (`house_prices`,
OpenML data id 42165) — **1,460 houses × 79 explanatory features** describing
residential homes in Ames, Iowa, with `SalePrice` as the target. Loaded directly
in-notebook via `sklearn.datasets.fetch_openml`, so no manual download is required.

- **19 columns** had missing values, ranging from 0.07% (`Electrical`) to 99.5%
  (`PoolQC` — meaning "no pool," not unknown data)
- After one-hot encoding categorical features: **272 numerical features**
- Train/test split: **1,168 training rows / 292 testing rows** (80/20)

## How to Run

1. Open `housing_regression.ipynb` in [Google Colab](https://colab.research.google.com/)
   (or JupyterLab locally).
2. Run all cells top to bottom — the dataset is fetched automatically from OpenML.
3. No API keys, credentials, or manual file uploads needed.

To run locally instead of Colab:
```bash
pip install -r requirements.txt
jupyter notebook housing_regression.ipynb
```

## Known Simplifications / Future Improvements

- Missing-value imputation and one-hot encoding are fit on the full dataset before
  the train/test split (a common simplification for this scale of project); a
  stricter pipeline would fit these on the training set only, as is already done
  for `StandardScaler`.
- Linear Regression's negative R² strongly suggests multicollinearity from one-hot
  encoding; **Ridge or Lasso regression** would regularize this and should perform
  far closer to the tree-based models.
- `SalePrice` is right-skewed; a log transform could improve Linear Regression
  performance further.
- No hyperparameter tuning yet — `GridSearchCV` / `RandomizedSearchCV` on the
  Random Forest and Decision Tree would likely push R² above 0.89.
- Gradient boosting models (XGBoost, LightGBM) are a natural next comparison point.

## License

This project is released under the [MIT License](LICENSE).