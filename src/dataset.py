import pandas as pd
from pprint import pprint

from preprocessing import basic_cleaning, make_target_local_extremes
from features import add_lag_features, add_rolling_features
from features import add_ratio_features, add_time_features, add_bollinger_bands


def build_features_for_token(
    df: pd.DataFrame,
    price_col: str = 'daily_closing_price_usd',
    volume_col: str = 'daily_trading_volume_usd',
    addresses_col: str = 'daily_active_addresses',
    horizon: int = 7
) -> pd.DataFrame:


    df_clean = basic_cleaning(df)
    df_target = make_target_local_extremes(df_clean,
                                           price_col=price_col,
                                           horizon=horizon)

    # lags
    df_feats = add_lag_features(
        df_target,
        cols_to_lag=[price_col, volume_col, addresses_col], 
        lags=[1,2,3,7])
    
    # means
    df_feats = add_rolling_features(df_feats, col=price_col, windows=[3,7])
    
    df_feats = add_ratio_features(
        df_feats,
        numerator_cols=[addresses_col, volume_col], 
        denominator_cols=[price_col])
    
    # time features
    df_feats = add_time_features(df_feats)
    
    df_feats = add_bollinger_bands(df_feats, price_col=price_col, window=20, num_std=2.0)
    # drop new nan
    df_feats.dropna(inplace=True)

    return df_feats


def build_features_for_all_tokens(data_dict: dict, horizon=7) -> dict:
    """
      { token_name: df_final }
    """
    all_tokens_features = {}
    for token_name, df_token in data_dict.items():
        print(f"Building features for {token_name} ...")
        df_with_features = build_features_for_token(
            df_token,
            price_col='daily_closing_price_usd',
            volume_col='daily_trading_volume_usd',
            addresses_col='daily_active_addresses',
            horizon=horizon
        )
        all_tokens_features[token_name] = df_with_features
    return all_tokens_features


if __name__ == '__main__':
    data_dict = pd.read_pickle('../data/san_tokens_PA_data_100.pkl')
    all_prepared_data = build_features_for_all_tokens(data_dict, horizon=7)
    pprint(all_prepared_data)