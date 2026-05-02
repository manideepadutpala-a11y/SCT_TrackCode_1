# SCT_TrackCode_1

## House Prices Linear Regression Task

This project contains an implementation of house price regression using the Kaggle Ames Housing dataset.

### Files

- `house_prices_linear_regression.py`: Training and prediction script with both basic and advanced feature sets, plus a gradient boosting option.
- `requirements.txt`: Required Python packages.

### Model details

The script supports:
- `basic` feature set: `GrLivArea`, `BedroomAbvGr`, `TotalBathrooms`
- `advanced` feature set: quality, size, garage, basement, porch, age, and categorical encodings
- `linear` model: ordinary least squares
- `gradient_boosting` model: `HistGradientBoostingRegressor`

### Usage

1. Put `train.csv` and `test.csv` from the Kaggle dataset into this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training and prediction:

```bash
python house_prices_linear_regression.py --train train.csv --test test.csv --output submission.csv --model gradient_boosting --feature-set advanced --save-model house_price_model.pkl
```

4. Run only training and cross-validation:

```bash
python house_prices_linear_regression.py --train train.csv --model gradient_boosting --feature-set advanced --save-model house_price_model.pkl
```

5. Start the Flask app and open the interactive dashboard in your browser:

```bash
python app.py
```

Then visit:

```text
http://127.0.0.1:5000
```

### Notes

- The advanced feature set uses engineered values such as `TotalPorchSF`, `TotalSF`, `Age`, and `RemodelAge`.
- The gradient boosting model generally performs better than linear regression on this dataset.
