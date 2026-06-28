import sys
import traceback

import fdb
# This program prints the content of the Multicharts Databases.
# The version of FireBird used by Multicharts is 11 and the latest Python drivers required version 12.T
# The first step is to download this package https://github.com/jaclas/Firebird-ODS-11-to-12-converter/tree/main
# and run the batch file of each of the 3 Multicharts databases to convert them to 12.
# Also the Firebird client library must be installed and the path to the fbclient.dll must be set in the FB_CLIENT variable below.
# Latest version can be downloaded from here https://www.firebirdsql.org/en/firebird-3-0/
# The installation directory need to be added to the PATH environment variable.
# FBPORTFOLIO_12.GDB is the database that contains multichart configuration like exchanges, currencies and holidays.
# The other 2 databases contain the historical data but it is in blob format.

DB_STORAGE_PATH = r'C:\Invest\Databases\TSSTORAGE_12.GDB'
DB_CACHE_PATH = r'C:\Invest\Databases\TSCACHE_12.GDB'
DB_PORTFOLIO_PATH = r'C:\Invest\Databases\FBPORTFOLIO_12.GDB'

FB_CLIENT = r'C:\bin\Firebird-3.0.13.33818-0-x64\fbclient.dll'
SQL_LIST_TABLES = """
    SELECT COALESCE(TRIM(TRAILING FROM rdb$relation_name), '')
    FROM rdb$relations
    WHERE rdb$view_blr IS NULL
      AND (rdb$system_flag = 0 OR rdb$system_flag IS NULL)
    ORDER BY 1
"""

SQL_BLOB_DATA = """
SELECT * FROM EXCHANGES
"""

def main():
    try:
        fdb.load_api(FB_CLIENT)
    except Exception as e:
        print("Failed to load FB client:", e, file=sys.stderr)
        return

    try:
        # For embedded, pass the file path as the dsn and provide fb_library_name
        conn = fdb.connect(dsn=DB_PORTFOLIO_PATH, user='SYSDBA', password='masterkey',
                           fb_library_name=FB_CLIENT)
    except Exception as e:
        print("Connection failed:", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return

    try:
        cur = conn.cursor()
        try:
            cur.execute(SQL_LIST_TABLES)
            rows = cur.fetchall()
            for row in rows:
                relation_name = row[0]
                if relation_name is not None:
                    print(relation_name.strip())

            cur.execute(SQL_BLOB_DATA)
            rows = cur.fetchall()

            # get column names from cursor.description
            cols = [col[0].strip() if isinstance(col[0], str) else col[0] for col in cur.description]
            # print a header line
            print(' | '.join(cols))

            # print each row, converting BLOB/bytes and None to readable strings
            for row in rows:
                out = []
                for val in row:
                    if isinstance(val, (bytes, bytearray)):
                        out.append(f'<BLOB {len(val)} bytes>')
                    elif val is None:
                        out.append('NULL')
                    else:
                        out.append(str(val))
                print(' | '.join(out))

        finally:
            try:
                cur.close()
            except Exception:
                pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
