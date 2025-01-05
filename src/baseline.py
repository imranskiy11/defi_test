# from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from dataset import build_features_for_all_tokens

import warnings
warnings.filterwarnings('ignore')


def split_for_training(df: pd.DataFrame,
                       target_cols=['delta_min', 'delta_max']):
    # featurrs
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols]
    y = df[target_cols]
    return X, y


def time_series_split(df, train_ratio=0.8, val_ratio=0.1):
    """
    """
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    return train_df, val_df, test_df


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100.0 * np.mean(
        2.0 * np.abs(y_pred - y_true) / 
        (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )


def time_series_cv_eval(model, X, y, n_splits=5):
    """
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_list, mae_list, r2_list = [], [], []
    mape_list, smape_list = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model_fold = lgb.LGBMRegressor(**model.get_params())
        
        model_fold.fit(X_train_fold, y_train_fold)

        y_pred = model_fold.predict(X_val_fold)

        rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
        mae = mean_absolute_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)
        mape = mean_absolute_percentage_error(y_val_fold, y_pred)
        smape = symmetric_mean_absolute_percentage_error(y_val_fold, y_pred)

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        mape_list.append(mape)
        smape_list.append(smape)

        print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, "
              f"MAPE={mape:.2f}%, SMAPE={smape:.2f}%, R2={r2:.4f}")


def remove_constant_columns(df: pd.DataFrame):
    """
    """
    desc = df.describe().T
    const_cols = desc[desc['std'] == 0].index
    df = df.drop(columns=const_cols, errors='ignore')
    return df, const_cols


if __name__ == '__main__':
    print('<<<>>> Starting <<<>>>')

    data_dict = pd.read_pickle('../data/san_tokens_PA_data_100.pkl')
    all_prepared_data = build_features_for_all_tokens(data_dict, horizon=7)
    df_token = all_prepared_data['frax-share']
    train_df, val_df, test_df = time_series_split(df_token)
    target_cols = ['delta_min', 'delta_max']
    feature_cols = [c for c in df_token.columns if c not in target_cols]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    X_val = val_df[feature_cols]
    y_val = val_df[target_cols]

    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    X_train_full, dropped_cols = remove_constant_columns(X_train_full)
    X_test.drop(columns=dropped_cols, inplace=True, errors='ignore')

    print(f"Dropped constant columns: {list(dropped_cols)}")
    print(f"X_train_full.shape = {X_train_full.shape}")
    print(f"X_test.shape = {X_test.shape}")

    model_min_cv = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        random_state=42)

    print("\n>>> TimeSeries CV for delta_min <<<")
    results_cv_min = time_series_cv_eval(
        model_min_cv,
        X_train_full,
        y_train_full['delta_min'],
        n_splits=5
    )
    print("CV results (delta_min):", results_cv_min)

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'learning_rate': [0.01, 0.05],
        'n_estimators': [500, 1000],
        'max_depth': [3, 5]
    }

    model_base_min = lgb.LGBMRegressor(random_state=11)

    grid_search_min = GridSearchCV(
        estimator=model_base_min,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        verbose=1
    )

    grid_search_min.fit(X_train_full, y_train_full['delta_min'])

    print("\nBest params (delta_min):", grid_search_min.best_params_)
    print("CV RMSE (delta_min):", -grid_search_min.best_score_)

    best_params_min = grid_search_min.best_params_
    model_min = lgb.LGBMRegressor(**best_params_min)
    model_min.fit(X_train_full, y_train_full['delta_min'])

    y_pred_min = model_min.predict(X_test)

    rmse_min = mean_squared_error(
        y_test['delta_min'], y_pred_min, squared=False)
    mae_min = mean_absolute_error(
        y_test['delta_min'], y_pred_min)
    r2_min = r2_score(y_test['delta_min'], y_pred_min)
    mape_min = mean_absolute_percentage_error(
        y_test['delta_min'], y_pred_min)
    smape_min = symmetric_mean_absolute_percentage_error(
        y_test['delta_min'], y_pred_min)

    print("\n<<< Final TEST metrics for delta_min after tuning >>>")
    print(f"RMSE={rmse_min:.4f}, MAE={mae_min:.4f}, R2={r2_min:.4f}")
    print(f"MAPE={mape_min:.2f}%, SMAPE={smape_min:.2f}%")
