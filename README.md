# Customer Churn Analysis

Exploratory data analysis, feature engineering, and a random forest churn model for an energy utility's small and medium business customers.

The notebooks come from the BCG data science virtual experience program, where the client, an energy utility, is called PowerCo.
The question is whether customer churn is driven by price sensitivity, and whether churn can be predicted from customer and pricing data.

## Data

Four CSV files are committed to the repository, so the notebooks run without any external download.

1. `client_data.csv` has 14,606 customers in 26 columns, with no missing values.
The columns cover consumption (`cons_12m`, `cons_gas_12m`, `cons_last_month`, `imp_cons`), forecasts (`forecast_cons_12m`, `forecast_price_energy_off_peak`, and similar), margins (`margin_gross_pow_ele`, `margin_net_pow_ele`, `net_margin`), contract dates (`date_activ`, `date_end`, `date_modif_prod`, `date_renewal`), categorical fields (`channel_sales`, `origin_up`, `has_gas`), and the target `churn`.
2. `price_data.csv` has 193,002 rows in 8 columns, also with no missing values.
It holds monthly energy and power prices per customer for January to December 2015, split into off-peak, peak and mid-peak, each with a variable and a fixed part.
Most companies have all 12 monthly rows, and a small number have between 7 and 11.
3. `churn_data_modeling.csv` is the merged table of client data and price features that the EDA notebook writes at the end, and it is the input to the modelling notebook.
4. `out_of_sample_data_with_predictions.csv` holds the 3,652 held-out test rows with the model's predicted `churn` label and `churn_probability`.

The values in `channel_sales` and `origin_up` are hashed strings in the source data, so the categories are anonymous.

## Notebooks

### `Exploratory_Data_Analysis_BCG.ipynb`

This notebook loads the client and price data, converts the date columns to datetime, and checks shapes, descriptive statistics and missing values.
It then plots the churn split, the consumption and forecast distributions by churn, and the distributions of the six price columns.
For price sensitivity, it averages each price column per customer over the full year, the last 6 months and the last 3 months, and adds combined off-peak, peak and mid-peak prices (variable plus fixed).
Those price features are correlated against churn, and finally the client data is merged with them and written out as `churn_data_modeling.csv`.

### `Feature_engineering_BCG.ipynb`

Starts from `churn_data_modeling.csv` and adds more features:

- `offpeak_diff_dec_january_energy` and `offpeak_diff_dec_january_power`, the December minus January off-peak price difference per customer.
- Mean differences between off-peak, peak, and mid-peak prices, for both the variable and fixed components.
- The maximum monthly value of each of those differences per customer.
- `tenure`, the whole number of years between `date_activ` and `date_end`.
- `months_activ`, `months_to_end`, `months_modif_prod`, and `months_renewal`, measured against a reference date of 2016-01-01, after which the raw date columns are dropped.
- `has_gas` converted from `t`/`f` to 1/0, and one-hot encoding of `channel_sales` and `origin_up` with the rarest categories dropped.
- Log base 10 transforms on the skewed consumption and forecast columns.
- `num_years_antig` and `forecast_cons_year` dropped because they correlate highly with other features.

The final feature matrix is 14,605 rows by 71 columns.
It is split 75/25 into 10,953 training rows and 3,652 test rows with `random_state=42`, and a `RandomForestClassifier` with 1,000 trees is fitted on the training part.
The notebook plots the feature importances and writes the test set predictions and probabilities to `out_of_sample_data_with_predictions.csv`.

## Findings

These are the results recorded in the notebooks.

1. Churn is imbalanced: 90.28 percent of customers were retained and 9.72 percent churned.
2. The engineered price features have a very low correlation with churn, so the analysis concludes that price changes are not the main explanation for churn.
3. Churn rate falls as tenure grows.
It is highest at a tenure of 3 years (14.4 percent), 2 years (13.3 percent) and 4 years (12.6 percent), and lowest at 10 years (2.0 percent) and 9 years (2.4 percent).
4. Customers who also buy gas churn less, at 8.19 percent for `has_gas = 1` against 10.05 percent for `has_gas = 0`.
5. On the test set the random forest scores 0.9028 accuracy, 1.0 precision and 0.0483 recall.
The confusion matrix is 18 true positives, 0 false positives, 3,279 true negatives and 355 false negatives.
That accuracy is about what you would get by predicting "no churn" for everyone (90.28 percent), and with recall that low the model finds almost none of the customers who actually churn.
An untuned classifier on an imbalanced target usually behaves like this, and fixing it is the first thing to do before the model is useful.

## Running the notebooks

The notebooks were written in Google Colab, so a few `pd.read_csv` calls use `/content/` paths.
Change those to the local filenames when running outside Colab.

Requirements are pandas, numpy, seaborn, matplotlib, and scikit-learn.

```
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
jupyter notebook
```

Run `Exploratory_Data_Analysis_BCG.ipynb` first, since it produces `churn_data_modeling.csv`, and then `Feature_engineering_BCG.ipynb`.
A copy of `churn_data_modeling.csv` is already committed, so the modelling notebook can also be run on its own.

Cell outputs have been stripped from both notebooks to keep the repository small, so re-run them to regenerate the plots and printed results.
