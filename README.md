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

*(Fill in after running the notebook — copy the values from the Section 11
comparison table.)*

| Model                    | MAE | MSE | RMSE | R² Score |
|--------------------------|-----|-----|------|----------|
| Random Forest Regressor  |     |     |      |          |
| Decision Tree Regressor  |     |     |      |          |
| Linear Regression        |     |     |      |          |

**Best model:** *(fill in)*, chosen for the highest R² / lowest RMSE on the held-out
test set.

## Tech Stack

- Python
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — modeling, preprocessing, evaluation

## Dataset

[Ames Housing dataset](https://www.openml.org/search?type=data&id=42165) (`house_prices`,
OpenML data id 42165) — 79 explanatory features describing residential homes in
Ames, Iowa, with `SalePrice` as the target. Loaded directly in-notebook via
`sklearn.datasets.fetch_openml`, so no manual download is required.

## How to Run

1. Open `Ames_Housing_Regression.ipynb` in [Google Colab](https://colab.research.google.com/)
   (or JupyterLab locally).
2. Run all cells top to bottom — the dataset is fetched automatically from OpenML.
3. No API keys, credentials, or manual file uploads needed.

To run locally instead of Colab:
```bash
pip install -r requirements.txt
jupyter notebook Ames_Housing_Regression.ipynb
```

## Known Simplifications / Future Improvements

- Missing-value imputation and one-hot encoding are fit on the full dataset before
  the train/test split (a common simplification for this scale of project); a
  stricter pipeline would fit these on the training set only, as is already done
  for `StandardScaler`.
- `SalePrice` is right-skewed; a log transform could improve Linear Regression
  performance in particular.
- No hyperparameter tuning yet — `GridSearchCV` / `RandomizedSearchCV` on the
  Random Forest and Decision Tree would likely improve results further.
- Gradient boosting models (XGBoost, LightGBM) are a natural next comparison point.

## License

This project is released under the [MIT License](LICENSE).
