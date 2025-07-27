from enum import Enum

OUTPUT_DIR = "/Invest/research/yahoo"

class MarketHighlight(Enum):
    TOP_GAINERS = ("gainers","get_day_gainers")
    TOP_LOSERS = ("losers", "get_day_losers")
    MOST_ACTIVE = ("most-active", "get_day_most_active")

    def __new__(cls, value, method):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.method = method
        return obj

class Exchanges(Enum):
    NASDAQ = ("nasdaq","tickers_nasdaq",{"Symbol":"Symbol Name","Security Name":"Description"},{"Category":"Stocks","Exchange":"NASDAQ"})
    DOW = ("dow", "tickers_dow",{"Symbol":"Symbol Name","Company": "Description"},{"Category":"Stocks"})
    SP500 = ("sp500", "tickers_sp500",{"Symbol":"Symbol Name","Security":"Description"},{"Category":"Stocks","Exchange":""})

    def __new__(cls, value, method,col_rename,col_add):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.method = method
        obj.col_rename = col_rename
        obj.col_add = col_add
        return obj

