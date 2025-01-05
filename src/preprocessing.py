import pandas as pd


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    df = df.copy()
    df.reset_index(inplace=True)
    df.drop_duplicates(subset='timestamp', inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True, drop=True)

    df.fillna(method='ffill', inplace=True)

    return df


def make_target_local_extremes(
        df: pd.DataFrame, price_col: str = 'daily_closing_price_usd',
        horizon: int = 7) -> pd.DataFrame:
    """
        delta_min = (min_price_in_next_horizon - current_price) / current_price * 100
        delta_max = (max_price_in_next_horizon - current_price) / current_price * 100
        horizon - horizon days
    """
    df = df.copy()
    current_price = df[price_col]

    min_price_forward = current_price.shift(-1).rolling(window=horizon).min()
    max_price_forward = current_price.shift(-1).rolling(window=horizon).max()
    # calculate deltas
    df['delta_min'] = (min_price_forward - current_price) / current_price * 100
    df['delta_max'] = (max_price_forward - current_price) / current_price * 100

    df.dropna(subset=['delta_min', 'delta_max'], inplace=True)
    return df


def filter_anomalies(df: pd.DataFrame, 
                     price_col: str = 'daily_closing_price_usd',
                     addresses_col: str = 'daily_active_addresses',
                     z_threshold: float = 3.0,
                     price_spike_pct: float = 10.0) -> pd.DataFrame:
    """
    FIlter anomalies
    return df only with anomalies
    """
    df = df.copy()

    df['price_return_pct'] = df[price_col].pct_change() * 100
    # filter by price jump
    cond_price_spike = df['price_return_pct'].abs() > price_spike_pct
    # anomaly by active_addresses - z-score
    if addresses_col in df.columns:
        mean_addr = df[addresses_col].mean()
        std_addr = df[addresses_col].std()
        df['addresses_zscore'] = (df[addresses_col] - mean_addr) / std_addr
        cond_addr_spike = df['addresses_zscore'].abs() > z_threshold
    else:
        cond_addr_spike = False
    cond_final = cond_price_spike | cond_addr_spike
    df_anomalies = df[cond_final].copy()
    df_anomalies.drop(columns=['price_return_pct', 'addresses_zscore'],
                      errors='ignore', inplace=True)
    return df_anomalies


def preprocess_and_create_targets(df: pd.DataFrame,
                                  horizon_days: int = 7,
                                  price_col: str = 'daily_closing_price_usd'):
    """
    """
    df_clean = basic_cleaning(df)

    df_target = make_target_local_extremes(
        df_clean,
        price_col=price_col,
        horizon=horizon_days)
    return df_target


def preprocess(data_dict: dict, horizon_days: int = 7,
               price_col: str = 'daily_closing_price_usd',
               addresses_col: str = 'daily_active_addresses',
               z_threshold: float = 3.0, price_spike_pct: float = 10.0,
               verbose: bool = True):
    """
    preprocessing + create a target (delta_min, delta_max).
    Filter abnormal days (by price or by daily_active_addresses).
    - all_tokens_data[token_name] = DataFrame (after preprocessing and target)
    - all_tokens_anomalies[token_name] = DataFrame (only anomaly rows)

    Parameters:
    - horizon_days: how many days ahead to look for calculating local min/max
    - price_col: name of the column with price (for target and anomalies)
    - addresses_col: name of column with daily_active_addresses (for anomalies)
    - z_threshold: z-score threshold for addresses_col
    - price_spike_pct: % price change threshold per day, above which 
      we consider an anomaly
    - verbose: whether to print information about the process
    """
    all_tokens_data = {}
    all_tokens_anomalies = {}

    for token_name, df_token in data_dict.items():
        if verbose:
            print(f"Processing {token_name} ...")

        df_cleaned = basic_cleaning(df_token)
        df_prepared = make_target_local_extremes(
            df_cleaned,
            price_col=price_col,
            horizon=horizon_days
        )

        df_anoms = filter_anomalies(
            df_prepared,
            price_col=price_col,
            addresses_col=addresses_col,
            z_threshold=z_threshold,
            price_spike_pct=price_spike_pct
        )

        all_tokens_data[token_name] = df_prepared
        all_tokens_anomalies[token_name] = df_anoms

    return all_tokens_data, all_tokens_anomalies
