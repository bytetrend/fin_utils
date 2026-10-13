import optuna
from optuna import Study

def warm_start_optuna(df_historical, target_params):
    """Creates a study and populates it with your 500 existing spreadsheet runs."""
    study: Study = optuna.create_study(direction="maximize")

    for _, row in df_historical.iterrows():
        # 1. Define active dynamic search distributions for your target features
        distributions = {
            "min_n": optuna.distributions.IntDistribution(10, 100),
            "test_fraction": optuna.distributions.FloatDistribution(0.1, 0.5)
        }

        # 2. Extract values dynamically based on what columns the LLM requested
        params_dict = {
            "min_n": int(row.get("min_n", 30)),
            "test_fraction": float(row.get("test_fraction", 0.25))
        }

        # Add the target indicator column ranges dynamically
        for col in target_params:
            if col in row:
                distributions[col] = optuna.distributions.FloatDistribution(
                    float(df_historical[col].min()),
                    float(df_historical[col].max())
                )
                params_dict[col] = float(row[col])

        # 3. Handle data records missing target performance metrics gracefully
        target_value = float(row.get("Profit/Loss", 0.0))

        # 4. Synthesize the trial historical record
        trial = optuna.trial.create_trial(
            params=params_dict,
            distributions=distributions,
            value=target_value
        )
        study.add_trial(trial)

    return study