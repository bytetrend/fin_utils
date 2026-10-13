import csv
import glob
import os
import shutil
from datetime import datetime

LOG_SIGNAL_DIR = r"C:\Invest\logs\signal"
LOG_IND_DIR = r"C:\Invest\logs\ind"
OUT_DIR = r"C:\Invest\logs\merged"
TEMP_OUT_DIR = r"C:\Invest\logs\merged_tmp"
STRATEGIES = ["AtsSlowReversal", "AtsFastReversal"]

os.makedirs(TEMP_OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def sanitize_fieldnames(fieldnames):
    cleaned = []
    for f in fieldnames or []:
        if f is None:
            continue
        name = str(f).strip()
        if name:
            cleaned.append(name)
    return cleaned


def norm_date_from_indicator(s):
    s = (s or '').strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except Exception:
            pass
    return s


def norm_time_from_indicator(s):
    s = (s or '').strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%H%M"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M")
        except Exception:
            pass
    # fallback: take first 5 chars
    return s[:5]


def norm_date_from_trade(s):
    s = (s or '').strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except Exception:
            pass
    return s


def norm_time_from_trade(s):
    s = (s or '').strip()
    if ':' in s:
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(s, fmt).strftime("%H:%M")
            except Exception:
                pass
    else:
        s2 = s.zfill(4)
        try:
            return datetime.strptime(s2, "%H%M").strftime("%H:%M")
        except Exception:
            pass
    return s[:5]


def load_indicators(strategy):
    pattern = os.path.join(LOG_IND_DIR, f"{strategy}-*.csv")
    files = glob.glob(pattern)
    # best[key] = (has_rt1, tick, computertime, row)
    best = {}
    fieldnames = None
    for f in files:
        # Extract symbol from filename: Strategy-Symbol-*.csv
        fname = os.path.basename(f)
        parts = fname.split('-')
        symbol = parts[1] if len(parts) > 1 else ''

        with open(f, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is not None:
                cleaned = sanitize_fieldnames(reader.fieldnames)
                if fieldnames is None:
                    fieldnames = cleaned
            for row in reader:
                row = {k: v for k, v in row.items() if k is not None and str(k).strip()}
                rt_raw = (row.get('R/T') or '').strip()
                has_rt1 = rt_raw in ('1', '1.0', 'True')

                bdate = norm_date_from_indicator(row.get('BarDate', ''))
                btime = norm_time_from_indicator(row.get('BarTime', ''))
                bnum = (row.get('BarNumber') or '').strip()
                key = (symbol, bdate, btime, bnum)

                tick_raw = row.get('Tick', '')
                try:
                    tick_v = float(tick_raw)
                except Exception:
                    tick_v = float('inf')

                comp = row.get('computertime', '')
                try:
                    comp_v = float(comp)
                except Exception:
                    comp_v = 0.0

                prev = best.get(key)
                if prev is None:
                    best[key] = (has_rt1, tick_v, comp_v, row)
                else:
                    prev_has_rt1, prev_tick, prev_comp, prev_row = prev
                    replace = False
                    # Prefer rows with R/T == 1
                    if has_rt1 and not prev_has_rt1:
                        replace = True
                    elif has_rt1 == prev_has_rt1:
                        # lower tick wins
                        if tick_v < prev_tick:
                            replace = True
                        elif tick_v == prev_tick:
                            # tie-breaker: prefer latest computertime
                            if comp_v >= prev_comp:
                                replace = True
                    if replace:
                        best[key] = (has_rt1, tick_v, comp_v, row)
    # return mapping key->row (selected) and indicator fieldnames
    return {k: v[3] for k, v in best.items()}, (fieldnames or [])


def process_strategy(strategy):
    indicators, ind_fields = load_indicators(strategy)
    trades_pattern = os.path.join(LOG_SIGNAL_DIR, f"{strategy}-*-trades.csv")
    trade_files = glob.glob(trades_pattern)
    merged_rows = []
    missing_rows = []
    trade_fields = None
    for tf in trade_files:
        # Extract symbol from filename: Strategy-Symbol-*-trades.csv
        fname = os.path.basename(tf)
        parts = fname.split('-')
        symbol = parts[1] if len(parts) > 1 else ''
        
        with open(tf, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is not None:
                cleaned = sanitize_fieldnames(reader.fieldnames)
                if trade_fields is None:
                    trade_fields = cleaned
            for tro in reader:
                tro = {k: v for k, v in tro.items() if k is not None and str(k).strip()}
                edate = norm_date_from_trade(tro.get('EntryDate', ''))
                etime = norm_time_from_trade(tro.get('EntryTime', ''))
                sbar = (tro.get('SignalBar') or '').strip()
                key = (symbol, edate, etime, sbar)
                ind = indicators.get(key)
                # Fallback: if no exact time match, try matching by symbol/date/bar number and ignore time.
                if not ind:
                    ind = next((row for (sym, bdate, btime, bnum), row in indicators.items()
                                if sym == symbol and bdate == edate and bnum == sbar), None)
                merged = dict(tro)
                if ind:
                    for k, v in ind.items():
                        if k is None or not str(k).strip():
                            continue
                        merged[f'ind_{k}'] = v
                else:
                    # still add ind_ fields as empty to keep columns consistent
                    for k in sanitize_fieldnames(ind_fields):
                        merged[f'ind_{k}'] = ''
                    trade_row = {field: tro.get(field, '') for field in sanitize_fieldnames(trade_fields)}
                    missing_rows.append({"filename": fname, **trade_row})
                merged_rows.append(merged)
    if trade_fields is None:
        print(f"No trade files found for strategy {strategy}")
        return
    # build output fieldnames: trades fields then ind_ fields
    out_fields = sanitize_fieldnames(trade_fields)
    # ensure ind fields unique and in original order
    for f in sanitize_fieldnames(ind_fields):
        pref = f'ind_{f}'
        if pref not in out_fields:
            out_fields.append(pref)
    out_path = os.path.join(TEMP_OUT_DIR, f"{strategy}-merged.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as outfh:
        writer = csv.DictWriter(outfh, fieldnames=out_fields, extrasaction='ignore')
        writer.writeheader()
        for r in merged_rows:
            writer.writerow(r)
    print(f"Wrote {len(merged_rows)} merged rows to {out_path}")

    missing_out_path = os.path.join(TEMP_OUT_DIR, f"{strategy}-mising-signal.csv")
    missing_fields = ['filename'] + sanitize_fieldnames(trade_fields)
    with open(missing_out_path, 'w', newline='', encoding='utf-8') as outfh:
        writer = csv.DictWriter(outfh, fieldnames=missing_fields, extrasaction='ignore')
        writer.writeheader()
        for row in missing_rows:
            writer.writerow(row)
    print(f"Wrote {len(missing_rows)} missing-indicator rows to {missing_out_path}")


if __name__ == '__main__':
    for s in STRATEGIES:
        process_strategy(s)
    # Move temp files to final location
    for s in STRATEGIES:
        for suffix in ["-merged.csv", "-mising-signal.csv"]:
            src = os.path.join(TEMP_OUT_DIR, f"{s}{suffix}")
            dst = os.path.join(OUT_DIR, f"{s}{suffix}")
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"Moved {src} to {dst}")
    print('Done')
