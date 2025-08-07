import pandas as pd
import numpy as np
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.metrics import precision_recall_curve, auc
from itertools import product
from joblib import Parallel, delayed
from event_precision_recall import *
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.exceptions import UndefinedMetricWarning
import warnings


# === Utility Functions ===
def get_std_Brouwer(satellite):
    data = satellite.df_orbit['Brouwer mean motion']
    data = standardize(data)
    data.index = satellite.df_orbit.index
    return data

def get_diff_std_brouwer(satellite):
    data = satellite.df_orbit['Brouwer mean motion']
    diff_data = data.diff().dropna() # first-order-differencing
    scaler = StandardScaler()
    std_diff = scaler.fit_transform(diff_data.values.reshape(-1, 1)).flatten()
    return pd.Series(std_diff, index=diff_data.index)

def create_manoeuvre_buffer(df, buffer):
    df['actual_manoeuvre_buffered'] = 0
    man_dates = df[df['actual_manoeuvre'] == 1].index
    for date in man_dates:
        start = date - timedelta(days=buffer)
        end = date + timedelta(days=buffer)
        df.loc[(df.index >= start) & (df.index <= end), 'actual_manoeuvre_buffered'] = 1
    return df['actual_manoeuvre_buffered']

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def create_lag_features(series, n_lags):
    X, y, y_index = [], [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
        y_index.append(series.index[i])
    return np.array(X), np.array(y), pd.Index(y_index)

def diff_brouwer(satellite):
    df = satellite.df_orbit.copy()

    df['bmm_diff']       = df['Brouwer mean motion'].diff()

    df['bmm_diff_lag1']  = df['bmm_diff'].shift(1)   # ε_{t-1}
    df['bmm_diff_lag2']  = df['bmm_diff'].shift(2)   # ε_{t-2}

    df = df.dropna()

    X = df[['bmm_diff_lag1', 'bmm_diff_lag2']].values
    y = df['Brouwer mean motion'].values
    y_index = df.index

    return X, y, y_index


def add_features(satellite, n_lags):
    df = satellite.df_orbit.copy()


    df['mean_motion_diff'] = df['Brouwer mean motion'].diff()
    df['eccentricity_diff'] = df['eccentricity'].diff()
    df['inclination_diff'] = df['inclination'].diff()
    df['raan_diff'] = df['right ascension'].diff()


    df['mean_motion_rolling_mean'] = df['Brouwer mean motion'].rolling(window=n_lags).mean()
    df['mean_motion_rolling_std'] = df['Brouwer mean motion'].rolling(window=n_lags).std()
    df['mean_motion_rolling_range'] = df['Brouwer mean motion'].rolling(window=n_lags).max() - df['Brouwer mean motion'].rolling(window=n_lags).min()

    df['ecc_rolling_std'] = df['eccentricity'].rolling(window=n_lags).std()
    df['inc_rolling_std'] = df['inclination'].rolling(window=n_lags).std()
    df['raan_rolling_std'] = df['right ascension'].rolling(window=n_lags).std()

    df = df.dropna()

    lag_X, y, y_index = create_lag_features(df['Brouwer mean motion'], n_lags)


    df_features_aligned = df.loc[y_index]
    feature_cols = [
        'mean_motion_diff', 'mean_motion_rolling_mean', 'mean_motion_rolling_std', 'mean_motion_rolling_range',
        'eccentricity', 'eccentricity_diff', 'ecc_rolling_std',
        'inclination', 'inclination_diff', 'inc_rolling_std',
        'argument of perigee', 'mean anomaly',
        'right ascension', 'raan_diff', 'raan_rolling_std'
    ]
    additional_features = df_features_aligned[feature_cols].values

    X = np.hstack((lag_X, additional_features))
    return X, y, y_index

def mad_score(residuals):
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return 0.6745 * (residuals - med) / mad




# === ARIMA Model ===
class ARIMAModel:
    def __init__(self, satellite):
        self.name = "ARIMA"
        self.satellite = satellite
        self.data = get_std_Brouwer(satellite)
        self.fitted_model = None
        self.fitted = None
        self.residuals = None
        self.eval_results = None
        self.best_params = None

    def train(self, p, q, d):
        model = ARIMA(self.data, order=(p, q, d))
        self.fitted_model = model.fit()

        self.fitted = self.fitted_model.fittedvalues
        self.residuals = self.fitted_model.resid

        original_index = self.data.index

        if not isinstance(self.fitted.index, pd.DatetimeIndex):
            self.fitted.index = original_index[-len(self.fitted):]

        if not isinstance(self.residuals.index, pd.DatetimeIndex):
            self.residuals.index = original_index[-len(self.residuals):]


    def evaluate(self, buffer):
        df_ture = self.satellite.df_man
        
        series_truth = pd.to_datetime(df_ture['manoeuvre_date'])
            
        s_scores = pd.Series(
                        StandardScaler().fit_transform(self.residuals.to_frame()).flatten(),
                        index=self.residuals.index,
                        name="residual_zscore"
                                )
        
        y_true = s_scores.index.isin(series_truth).astype(int)

        _, _, thresholds = precision_recall_curve(y_true, s_scores)
        precisions = []
        recalls = []
        for thr in thresholds:
            p, r = compute_simple_matching_precision_recall_for_one_threshold(
                    matching_max_days=buffer,       
                    threshold=thr,
                    series_ground_truth_manoeuvre_timestamps=series_truth,
                    series_predictions=s_scores)
            precisions.append(p)
            recalls.append(r)

        pr_auc = auc(recalls, precisions)

        self.eval_results = {
            "buffer": buffer,
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": thresholds,
            "pr_auc": pr_auc
        }


    def train_and_evaluate_one_setting(self, p, q, d, buffer):
        model = ARIMAModel(self.satellite)
        model.train(p,q,d)
        model.evaluate(buffer)
        return {
            "p":p,
            "q":q,
            "d":d,
            "buffer": buffer,
            "pr_auc": model.eval_results["pr_auc"],
            "precisions": model.eval_results["precisions"],
            "recalls": model.eval_results["recalls"],
            "thresholds": model.eval_results["thresholds"],
            "model": model
        }


    def grid_search(self, param_grid, buffer):
        param_list = list(product(param_grid['p'],
                                  param_grid['q'],
                                  param_grid['d']))

        results = Parallel(n_jobs=-1)(
            delayed(self.train_and_evaluate_one_setting)(p, q, d, buffer)
            for (p, q, d) in param_list
        )
        results = [r for r in results if r is not None]
        if not results:
            raise ValueError("All parameter combinations failed.")

        # pick the best run
        best_res = max(results, key=lambda x: x['pr_auc'])
        best_model: ARIMAModel = best_res['model']

        # copy its state into this instance
        self.fitted_model = best_model.fitted_model
        self.fitted       = best_model.fitted
        self.residuals    = best_model.residuals

        # store the grid search summary
        self.eval_results       = best_res
        self.best_params        = {'p':best_res['p'],
                                   'q':best_res['q'],
                                   'd':best_res['d']}
        self.grid_search_results = pd.DataFrame([
            {k:v for k,v in r.items() if k != 'model'}
            for r in results
        ])

        # also attach to satellite
        self.satellite.arima = self




class XGBoostModel:
    def __init__(self, satellite, brouwer_only=True, diff_order=0):
        self.name = "XGBoost"
        self.satellite = satellite
        self.brouwer_only = brouwer_only
        # Number of differences to apply: 0 = no diff, 1 = first-order, 2 = second-order, etc.
        self.diff_order = diff_order
        self.model = None
        self.fitted = None
        self.residuals = None
        self.eval_results = None
        self.best_params = None
        self.grid_search_results = None

    def train(self, n_lags, n_estimators, max_depth, learning_rate, colsample_bytree):
        # 1) Load the Brouwer mean motion series
        series = get_std_Brouwer(self.satellite)  # pandas Series of Brouwer mean motion
        
        # 2) Apply differencing if requested
        if self.diff_order > 0:
            series = series.diff(self.diff_order)
        series = series.dropna()

        # 3) Create lag features for the series
        X, y, y_index = create_lag_features(series, n_lags)

        # 4) Initialize and fit the XGBoost regressor
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            random_state=42
        )
        self.model.fit(X, y)

        # 5) Store fitted values and compute residuals
        preds = self.model.predict(X)
        self.fitted = pd.Series(preds, index=y_index)
        y_series = pd.Series(y, index=y_index)
        self.residuals = y_series - self.fitted

    def evaluate(self, buffer):
        # Load true manoeuvre timestamps
        df_true = self.satellite.df_man
        truth_times = pd.to_datetime(df_true['manoeuvre_date'])

        # Use absolute residuals as anomaly scores
        scores = pd.Series(np.abs(self.residuals), index=self.residuals.index)

        # Build binary ground truth: 1 at timestamps with manoeuvres, else 0
        y_true = scores.index.isin(truth_times).astype(int)
        if y_true.sum() == 0:
            warnings.warn("No manoeuvre events in this segment. Skipping PR evaluation.", UndefinedMetricWarning)
            self.eval_results = {
                "buffer": buffer,
                "precisions": [],
                "recalls": [],
                "thresholds": [],
                "pr_auc": 0.0
            }
            return

        # Compute thresholds from precision–recall curve
        _, _, thresholds = precision_recall_curve(y_true, scores)

        # Compute precision and recall at each threshold using buffer-aware matching
        precisions, recalls = [], []
        for thr in thresholds:
            p, r = compute_simple_matching_precision_recall_for_one_threshold(
                matching_max_days=buffer,
                threshold=thr,
                series_ground_truth_manoeuvre_timestamps=truth_times,
                series_predictions=scores
            )
            precisions.append(p)
            recalls.append(r)

        # Compute area under the PR curve
        pr_auc = auc(recalls, precisions)
        self.eval_results = {
            'buffer': buffer,
            'precisions': precisions,
            'recalls': recalls,
            'thresholds': thresholds,
            'pr_auc': pr_auc
        }

    def grid_search(self, param_grid, buffer, n_jobs=-1):
        # Expect param_grid to include 'diff_order': list of int
        keys = ['n_lags', 'n_estimators', 'max_depth',
                'learning_rate', 'colsample_bytree', 'diff_order']

        # Build all combinations of hyperparameters
        param_list = list(product(
            param_grid['n_lags'],
            param_grid['n_estimators'],
            param_grid['max_depth'],
            param_grid['learning_rate'],
            param_grid['colsample_bytree'],
            param_grid['diff_order']
        ))

        def train_and_eval(n_lags, n_estimators, max_depth,
                           learning_rate, colsample_bytree, diff_order):
            # Instantiate a new model for each combination
            model = XGBoostModel(
                satellite=self.satellite,
                brouwer_only=self.brouwer_only,
                diff_order=diff_order
            )
            # Train and evaluate
            model.train(n_lags, n_estimators, max_depth, learning_rate, colsample_bytree)
            model.evaluate(buffer)
            return {
                'n_lags': n_lags,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'colsample_bytree': colsample_bytree,
                'diff_order': diff_order,
                'pr_auc': model.eval_results['pr_auc'],
                'model': model
            }

        # Run grid search in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_and_eval)(*params) for params in param_list
        )

        # Select the best result by PR-AUC
        best = max(results, key=lambda x: x['pr_auc'])
        self.best_params = {k: best[k] for k in keys}
        self.fitted = best['model'].fitted
        self.residuals = best['model'].residuals
        self.eval_results = best['model'].eval_results
        # Record all grid search outcomes (excluding model objects)
        self.grid_search_results = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'model'} for r in results
        ])
        self.satellite.xgb = self




# === Segmented Model for SARAL ===
def train_segmented_models_with_search(satellite, break_times, model_class, model_grid, buffer=3, model_kwargs=None):
    """
    For each segment (based on break_times), perform grid search and store the best model.
    Return:
        - segment_models: list of best model instances per segment
        - segment_ranges: list of (start, end) datetime tuples per segment
        - all_residuals: combined residuals from all segments
    """
    if model_kwargs is None:
        model_kwargs = {}

    all_times = [satellite.df_orbit.index.min()] + break_times + [satellite.df_orbit.index.max()]
    segment_models = []
    segment_ranges = []
    residual_list = []

    for i in range(len(all_times) - 1):
        start, end = all_times[i], all_times[i + 1]
        sub_sat = satellite.copy_range(start, end)
        segment_ranges.append((start, end))

        print(f"\n Segment {i+1}: {start.date()} ~ {end.date()}")

        # Grid search
        model = model_class(sub_sat, **model_kwargs)
        if model.name == "ARIMA":
            model.grid_search(model_grid, buffer)
        elif model.name == "XGBoost":
            model.grid_search(model_grid, buffer)
        else:
            raise ValueError("Unsupported model")

        segment_models.append(model)
        residual_list.append(model.residuals)

    all_residuals = pd.concat(residual_list).sort_index()
    return segment_models, segment_ranges, all_residuals

def evaluate_global_precision_recall(all_residuals, satellite, buffer):
    df_true = satellite.df_man
    series_truth = pd.to_datetime(df_true['manoeuvre_date'])

    s_scores = pd.Series(StandardScaler().fit_transform(all_residuals.to_frame()).flatten(),
                         index=all_residuals.index,
                         name="residual_zscore")

    y_true = s_scores.index.isin(series_truth).astype(int)

    _, _, thresholds = precision_recall_curve(y_true, s_scores)

    precisions, recalls = [], []
    for thr in thresholds:
        p, r = compute_simple_matching_precision_recall_for_one_threshold(
            matching_max_days=buffer,
            threshold=thr,
            series_ground_truth_manoeuvre_timestamps=series_truth,
            series_predictions=s_scores
        )
        precisions.append(p)
        recalls.append(r)

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    pr_auc = auc(recalls, precisions)

    return {
        "buffer": buffer,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
        "pr_auc": pr_auc
    }