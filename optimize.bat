
#C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\merge_trades.py

C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_performance_report.py C:\Invest\logs\merged\AtsFastReversal-merged.csv --excel C:\Invest\logs\merged\AtsFastReversal-report.xls
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_performance_report.py C:\Invest\logs\merged\AtsSlowReversal-merged.csv --excel C:\Invest\logs\merged\AtsSlowReversal-report.xls

C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_optuna_optimizer.py --output "C:\Invest\logs\merged\AtsFastReversal-optuna_optimized.json" "C:\Invest\logs\merged\AtsFastReversal-merged.csv"
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_feature_importance.py --output "C:\Invest\logs\merged\AtsFastReversal_feature_importance_report.json" "C:\Invest\logs\merged\AtsFastReversal-merged.csv"
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_param_optimizer.py --output "C:\Invest\logs\merged\AtsFastReversal-param_optimized.json" "C:\Invest\logs\merged\AtsFastReversal-merged.csv"
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_entryscore_weight_optimizer.py --n-trials 2000 --min-n 20 --max-weight 1 --cv-folds 5 --test-fraction 0.30 --min-threshold-frac 0.33 --output "C:\Invest\logs\merged\AtsFastReversal_entryscore_weights_optimizer.json" "C:\Invest\logs\merged\AtsFastReversal-merged.csv"

C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_optuna_optimizer.py --output "C:\Invest\logs\merged\AtsSlowReversal-optuna_optimized.json" "C:\Invest\logs\merged\AtsSlowReversal-merged.csv"
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_feature_importance.py --output "C:\Invest\logs\merged\AtsSlowReversal_feature_importance_report.json" "C:\Invest\logs\merged\AtsSlowReversal-merged.csv"
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_param_optimizer.py --output "C:\Invest\logs\merged\AtsSlowReversal-param_optimized.json" "C:\Invest\logs\merged\AtsSlowReversal-merged.csv"
C:\Invest\repo\fin_utils\.venv\Scripts\python.exe C:\Invest\repo\fin_utils\src\optimizer\ats_entryscore_weight_optimizer.py --n-trials 2000 --min-n 20 --max-weight 1 --cv-folds 5 --test-fraction 0.30 --min-threshold-frac 0.33 --output "C:\Invest\logs\merged\AtsSlowReversal_entryscore_weights_optimizer.json" "C:\Invest\logs\merged\AtsSlowReversal-merged.csv"