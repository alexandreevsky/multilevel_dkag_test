import pandas as pd


def filter_df_by_k_core(df: pd.DataFrame, source_col: str, target_col: str, k_core: int) -> pd.DataFrame:
    grouped = df.groupby(source_col)[target_col].nunique()
    filtered_sources = grouped[grouped >= k_core].index
    df = df[df[source_col].isin(filtered_sources)].reset_index(drop=True)
    return df
