import pandas as pd

from preprocessing import preprocess
from pprint import pprint


if __name__ == '__main__':

    data_dict = pd.read_pickle('../data/san_tokens_PA_data_100.pkl')
    all_data, all_anoms = preprocess(
        data_dict,
        horizon_days=7,
        price_col='daily_closing_price_usd',
        addresses_col='daily_active_addresses',
        z_threshold=3.0,
        price_spike_pct=10.0,
        verbose=True
    )

    pprint(all_anoms)