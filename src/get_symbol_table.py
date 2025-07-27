import os.path
import traceback
from datetime import date

import pandas as pd

from src.constants import Exchanges, OUTPUT_DIR
from yahoo_fin import stock_info as si


def get_ticker_table(sl: Exchanges) -> pd.DataFrame:
    try:
        print(f"Attempting to get day {sl.value}...")
        method_to_call = getattr(si, sl.method)
        return method_to_call(True)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    for ex in Exchanges:
        df: pd.DataFrame = get_ticker_table(ex)
        if df is not None:
            print(f"Shape: {df.shape}")
            df= df.rename(columns=ex.col_rename)
            for key,value in ex.col_add.items():
                df[key] = value
            df = df.loc[:,["Symbol Name","Description","Exchange","Category"]]
            out_directory = f'{OUTPUT_DIR}/symbols_table/{date.today().strftime("%Y%m%d")}'
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            df.to_csv(f"{out_directory}/{ex.method}.csv", header=True, index=False)
        else:
            print(f"Failed to get {ex.method} data.")
