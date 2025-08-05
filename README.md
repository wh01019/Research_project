# Satellite Manoeuvre Detection Framework

A Python toolkit for unsupervised detection of satellite manoeuvres from Brouwer Mean Motion time series. Leveraging ARIMA and XGBoost models with a modular, maintainable codebase.

---

## Features

- **Data Preparation**  
  • Load and parse orbital element CSVs and raw manoeuvre logs  
  • Normalize timestamps and slice by date ranges

- **Exploratory Data Analysis (EDA)**  
  • Standardized multi- and single-column plots with manoeuvre markers  
  • Configurable date clipping and subplot layouts

- **Modelling**  
  • **ARIMAModel**: Residual-based anomaly scoring, grid search, PR-AUC evaluation  
  • **XGBoostModel**: Lag-only or full-feature regression, residual scoring, grid search  
  • Segmented modelling for non-stationary regimes

- **Evaluation**  
  • Event-matching precision & recall (Zhao 2021) with configurable buffer days  
  • Standalone script for one-time PR calculation

---

## Repository Structure

## Repository Structure

```plaintext
.
├── datapreparation.py             # Data loaders & parsers
├── satellite_eda.py               # EDA plotting utilities
├── models.py                      # ARIMA & XGBoost model classes
├── event_precision_recall.py      # Precision/recall matching logic
├── requirements.txt               # Project dependencies
├── README.md                      # Documentation (this file)
└── examples/                      # Application notebooks
    ├── EDA.ipynb                  # Full EDA workflow
    ├── Evaluation-regular.ipynb   # PR evaluation on regular satellites
    ├── Evaluation-irregular-test.ipynb  # PR evaluation on irregular satellites
    └── SARAL_segmented_model.ipynb  # Segmented modelling example for SARAL
```


---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://your-repo-url.git
   cd your-repo
   ```

2. **Create & activate a Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Initialize Satellite**
```python
from datapreparation import Satellite
sat = Satellite("path/to/orbit.csv", "path/to/man.txt")
```

2. **Run EDA**
   ```python
   from satellite_eda import SatelliteEDA
   eda = SatelliteEDA(sat)
   eda.all_with_man(subplots_wrap=(3,2))
   ```
3. **Train & evaluate models**
   ```python
   from models import ARIMAModel, XGBoostModel

  # ARIMA grid search
  arima = ARIMAModel(sat)
  arima.grid_search({'p':[0,1], 'd':[0,1], 'q':[0,1]}, buffer=3)
  print(arima.best_params, arima.eval_results['pr_auc'])
  
  # XGBoost grid search
  xgb = XGBoostModel(sat)
  xgb.grid_search(
    {'n_lags':[3,5], 'n_estimators':[50,100], 'max_depth':[3,5],
     'learning_rate':[0.1], 'colsample_bytree':[0.8]},
    buffer=3
  )
  print(xgb.best_params, xgb.eval_results['pr_auc'])
   ```

## Code Quality & Organization

- **PEP8-compliant** and linted with `flake8`
- **Modular design**: clear separation of data I/O, EDA, modelling, and evaluation
- **Parallel grid search** using `joblib` for performance
- **Refactored utilities**: shared functions (`standardize`, `clip`, feature engineering) isolated for reuse

## Contributing

1. Fork and branch  
2. Add tests for new features  
3. Run `flake8` and fix any issues  
4. Submit a pull request  

## License

MIT © Your Name
