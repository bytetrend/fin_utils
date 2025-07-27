import os.path

from yahoo_fin import stock_info as si
import traceback
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import date

from src.constants import OUTPUT_DIR, MarketHighlight


def get_market_highlight(mh: MarketHighlight,count: int) -> pd.DataFrame:
    try:
        print(f"Attempting to get day {mh.value}...")
        method_to_call = getattr(si,mh.method)
        result: pd.DataFrame = method_to_call(count)
        print(f"Successfully retrieved {mh.value}")
        print(result)
        return result
    except KeyError as e:
        if "'52 Week Range'" in str(e):
            print("Known issue with '52 Week Range' column. Trying alternative approach...")
            return get_market_highlight_alternative(mh, count)
        else:
            print(f"KeyError: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

def get_market_highlight_alternative(mh:MarketHighlight, count: int)-> pd.DataFrame:
    try:
        print(f"Using alternative method to {mh.value}...")
        url = f"https://finance.yahoo.com/{mh.value}?offset=0&count={count}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) (en-US) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("Successfully fetched data from Yahoo Finance")
            
            # Parse HTML using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Find all tables on the page
            tables = soup.find_all('table')
            print(f"Found {len(tables)} tables on the page")
            
            for i, table in enumerate(tables):
                print(f"Table {i}: class={table.get('class', 'None')}")
                # Look for the first table with data
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has header + data rows
                    print(f"  - Has {len(rows)} rows")
                    # Check if this looks like a stock table
                    first_row = rows[0]
                    headers = [th.text.strip() for th in first_row.find_all(['th', 'td'])]
                    print(f"  - Headers: {headers}")
                    
                    # Look for stock-related headers
                    stock_keywords = ['Symbol', 'Name', 'Price', 'Change', 'Change %', 'Volume']
                    if any(keyword.lower() in ' '.join(headers).lower() for keyword in stock_keywords):
                        print(f"  - Table {i} looks like a stock table!")
                        
                        # Extract all data from this table
                        data = []
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all(['td', 'th'])
                            if cells:
                                row_data = [cell.text.strip() for cell in cells]
                                data.append(row_data)
                        
                        if data:
                            df = pd.DataFrame(data, columns=headers)
                            df = df.drop(columns=df.columns[2])
                            for i, row in df.iterrows():
                                row['Price'] =  row.to_dict()['Price'].split(" ")[0]
                            print("Successfully parsed day gainers data:")
                            print(df.head())
                            return df
            
            print("Could not find a suitable stock table")
            return None
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Alternative method failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    for mh in MarketHighlight:
        result: pd.DataFrame = get_market_highlight(mh,100)
        if result is not None:
            print(f"\nFinal result type: {type(result)}")
            if isinstance(result, pd.DataFrame):
                print(f"Shape: {result.shape}")
                out_directory = f'{OUTPUT_DIR}/day_actives/{date.today().strftime("%Y%m%d")}'
                if not os.path.exists(out_directory):
                    os.makedirs(out_directory)
                result.to_csv(f"{out_directory}/{mh.method}.csv")

        else:
            print(f"Failed to get {mh.method} data.")