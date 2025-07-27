import os.path
import traceback
from datetime import date
from typing import List

import pandas as pd

from src.constants import Exchanges, OUTPUT_DIR
from yahoo_fin import stock_info as si


def get_ticker_list(sl: Exchanges) -> List[str]:
    try:
        print(f"Attempting to get day {sl.value}...")
        method_to_call = getattr(si, sl.method)
        return method_to_call(False)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    for ex in Exchanges:
        result: List[str] = get_ticker_list(ex)
        if result is not None:
            out_directory = f'{OUTPUT_DIR}/symbols_list/{date.today().strftime("%Y%m%d")}'
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            with open(f"{out_directory}/{ex.method}.csv", "w") as f:
                f.write("Symbol\n")
                f.write("\n".join(result))

        else:
            print(f"Failed to get {ex.method} data.")
