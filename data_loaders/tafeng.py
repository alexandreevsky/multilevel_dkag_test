from pathlib import Path

import pandas as pd

from data_loaders.utils import filter_df_by_k_core


def load_tafeng_dataframe(user_k_core: int, item_k_core: int, data_folder_path: str = None) -> pd.DataFrame:
    if data_folder_path is None:
        file_path = Path(__file__).resolve().parent.parent / 'data' / 'ta_feng' / 'ta_feng_all_months_merged.csv'
    else:
        file_path = Path(f'{data_folder_path}/ta_feng/ta_feng_all_months_merged.csv')

    data = pd.read_csv(file_path).dropna().drop(['ASSET', 'AMOUNT', 'SALES_PRICE'], axis=1)
    data = data.reset_index(drop=True)
    data.columns = ['date', 'customer', 'age_group', 'zip_code', 'product_subclass', 'product']
    data.date = pd.to_datetime(data.date)
    for col in ['customer', 'age_group', 'zip_code', 'product_subclass', 'product']:
        data[col] = data[col].astype('str')

    data = filter_df_by_k_core(data, 'customer', 'product', user_k_core)
    data = filter_df_by_k_core(data, 'product', 'customer', item_k_core)

    return data
