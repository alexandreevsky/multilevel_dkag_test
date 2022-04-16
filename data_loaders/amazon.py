import ast
import gzip
import io
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
from toolz import curry, get

from data_loaders.utils import filter_df_by_k_core

AMAZON_URL_PREFIX = os.environ.get('AMAZON_DATA_URL')
META_COLUMNS = ['asin', 'category', 'also_buy', 'brand', 'also_view']
REVIEW_COLUMNS = ['reviewerID', 'asin', 'unixReviewTime']


def fetch(json_path: str) -> list:
    # /(metaFiles|categoryFiles)/json_name
    response = requests.get(url=f'{AMAZON_URL_PREFIX}/{json_path}')
    byte_stream = io.BytesIO(response.content)
    data = []
    with gzip.open(filename=byte_stream, mode='rt') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def fetch_meta_and_reviews(json_name: str) -> dict:
    return {
        'meta': fetch(f'metaFiles/meta_{json_name}.json.gz'),
        'reviews': fetch(f'categoryFiles/{json_name}.json.gz')
    }


def save_json(amazon_data: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(amazon_data, f)


@curry
def extract(fields: list, json_item: dict) -> tuple:
    return get(fields, json_item)


def to_frame(data: list, columns: list) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(map(extract(columns), data)), columns=columns)


def get_frames(amazon_data: dict) -> Dict[str, pd.DataFrame]:
    return {
        'reviews': to_frame(amazon_data['reviews'], REVIEW_COLUMNS),
        'meta': to_frame(amazon_data['meta'], META_COLUMNS)
    }


def load_amazon_dataframes(json_name: str, user_k_core: int, item_k_core: int, data_folder_path: str) -> Dict[str, pd.DataFrame]:
    if data_folder_path is None:
        data_dir = Path(__file__).resolve().parent.parent / 'data' / 'amazon' / json_name
    else:
        data_dir = Path(f'{data_folder_path}/amazon/{json_name}')

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

        frames = get_frames(fetch_meta_and_reviews(json_name))
        frames['reviews']['date'] = pd.to_datetime(frames['reviews'].unixReviewTime, unit='s')
        frames['reviews'] = frames['reviews'].rename({'reviewerID': 'customer', 'asin': 'product'}, axis=1)
        frames['meta'] = frames['meta'].rename({'asin': 'product'}, axis=1)

        frames['meta'].to_csv(data_dir / f'meta_{json_name}.csv', sep=';', index=False)
        frames['reviews'].to_csv(data_dir / f'{json_name}.csv', sep=';', index=False)
    else:
        frames = {
            'reviews': pd.read_csv(data_dir / f'{json_name}.csv', sep=';'),
            'meta': pd.read_csv(data_dir / f'meta_{json_name}.csv', sep=';')
        }
        frames['meta'].also_buy = frames['meta'].also_buy.map(ast.literal_eval)
        frames['meta'].also_view = frames['meta'].also_view.map(ast.literal_eval)
        frames['meta'].category = frames['meta'].category.map(ast.literal_eval)
        frames['reviews'].date = pd.to_datetime(frames['reviews'].date)

    frames['reviews'] = filter_df_by_k_core(frames['reviews'], 'customer', 'product', user_k_core)
    frames['reviews'] = filter_df_by_k_core(frames['reviews'], 'product', 'customer', item_k_core)
    unique_products_in_reviews = frames['reviews']['product'].unique()
    frames['meta'] = frames['meta'][frames['meta']['product'].isin(unique_products_in_reviews)]\
        .reset_index(drop=True)

    return frames
