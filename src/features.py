import pandas as pd
import numpy as np


def add_lag_features(
        df: pd.DataFrame,
        cols_to_lag: list,
        lags: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols_to_lag:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str,
                         windows: list = [3, 7]) -> pd.DataFrame:
    """
    """
    df = df.copy()
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(window=w).mean()
        df[f"{col}_std{w}"] = df[col].rolling(window=w).std()
    return df


def add_ratio_features(df: pd.DataFrame,
                       numerator_cols: list,
                       denominator_cols: list) -> pd.DataFrame:
    """
    """
    df = df.copy()
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns:
                ratio_name = f"ratio_{num_col}_to_{den_col}"
                # check div by zero
                df[ratio_name] = df[num_col] / (
                    df[den_col].replace(0, np.nan))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    df = df.copy()

    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df


def add_bollinger_bands(df: pd.DataFrame,
                        price_col: str = 'daily_closing_price_usd',
                        window: int = 20,
                        num_std: float = 2.0) -> pd.DataFrame:
    """
      upper_band = MA + num_std * std
      lower_band = MA - num_std * std
    """
    df = df.copy()
    ma_col = f"{price_col}_bb_ma{window}"
    std_col = f"{price_col}_bb_std{window}"
    df[ma_col] = df[price_col].rolling(window=window).mean()
    df[std_col] = df[price_col].rolling(window=window).std()

    df[f"{price_col}_bb_upper"] = df[ma_col] + num_std * df[std_col]
    df[f"{price_col}_bb_lower"] = df[ma_col] - num_std * df[std_col]

    return df


