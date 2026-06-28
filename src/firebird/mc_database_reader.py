# python
import subprocess
import shutil
import sys

# In this version the SQL command are run using an executable under the FireBird installation
# The output is grab by this python program. It is just to show another way to do it.
DB_PATH = r'C:\Invest\Databases\TSCACHE_12.GDB'
ISQL_PATH = shutil.which('isql') or r'C:\bin\Firebird-3.0.13.33818-0-x64\isql.exe'
USER = 'SYSDBA'
PASSWORD = 'masterkey'

SQL = """
SET HEADING OFF;
SET ECHO OFF;
SELECT rtrim(rdb$relation_name) FROM rdb$relations
 WHERE rdb$view_blr IS NULL
   AND (rdb$system_flag = 0 OR rdb$system_flag IS NULL)
 ORDER BY 1;
QUIT;
"""

if ISQL_PATH is None:
    print("isql not found. Install Firebird or set ISQL_PATH to the full path of `isql.exe`.", file=sys.stderr)
    sys.exit(1)

proc = subprocess.run(
    [ISQL_PATH, '-user', USER, '-password', PASSWORD, DB_PATH],
    input=SQL.encode('utf-8'),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

if proc.returncode != 0:
    print("isql failed:", proc.stderr.decode('utf-8', errors='replace'), file=sys.stderr)
    sys.exit(proc.returncode)

out = proc.stdout.decode('utf-8', errors='replace')
lines = [ln.strip() for ln in out.splitlines()]

# Filter out empty lines and typical isql prompts/notes
tables = [ln for ln in lines if ln and not ln.startswith('SQL>') and 'records affected' not in ln.lower()]

for t in tables:
    print(t)