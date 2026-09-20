## Commands to run
*** Check whether these carry real signal at all first (FeatureImportanceAnalyzer): ***
```commandline
python ats_feature_importance.py trades.csv --params ind_PipSpeedNorm,ind_PipSpeedTrendPct,ind_PipSpeedAcelNorm --cv-folds 5
```

Note: The `--params` flag accepts comparison-style tokens like `ind_X<5` or `ind_Y>=0.1`. For now these are parsed only to extract the column name on the left-hand side; the comparison is not applied as a row-level filter. If you want the CLI to apply the comparisons as dataset filters, request that change and it will be implemented.
(Worth noting: we already checked ind_PipSpeedNorm in the significance test and found p>0.3 for both directions on this batch — so whatever threshold comes out below should be treated as provisional, same caveat as everything else recently.)
Find the best single threshold on PipSpeedNorm (GridSearchOptimizer):
```commandline 
 ats_param_optimizer.py trades.csv --min-n 20 --top-n 8 --output param_report.json
 ```
Look at ind_PipSpeedNorm in the single_param_best section of the output — that's your data-driven PipSpeedLimit replacement. To test a specific candidate value directly:
```commandline
python ats_param_optimizer.py trades.csv --compare-filter "ind_PipSpeedNorm>=9" --direction short
```
Joint search across both real candidates at once (BayesianThresholdOptimizer):
```commandline
python ats_optuna_optimizer.py trades.csv --params ind_PipSpeedNorm,ind_PipSpeedTrendPct --n-trials 1000 --test-fraction 0.25 --min-n 20
```
Restricting --params to just these two (rather than searching everything) keeps the search honest given your current sample size, and gives you a genuine held-out test result rather than an in-sample number.
For PipSpeedFlipLimit, same command with ind_PipSpeedNorm, but treat the result as an approximation of C5's real behavior, not an exact calibration — since it's missing the [1]/[2] lag structure C5 actually uses.
If you want a precise PipSpeedFlipLimit recalibration (not just a proxy) or want to actually re-tune PipSpeedTrendPctLimit itself, that would need the strategy to log the lagged PipSpeed[1]/[2] values and the per-bar trend-window PipSpeed series respectively — neither is something I can back out from the entry-level trade log alone.
