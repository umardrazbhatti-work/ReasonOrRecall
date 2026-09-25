# Results — Kaggle run outputs, one folder per run

Drop each Kaggle run here in its own folder. Claude reads them from this folder.
The contents are gitignored (the repo is public and adapters are large); only
this README is tracked.

## Naming

`YYYY-MM-DD_<suite>_v<kaggle version>`, e.g. `2026-09-25_smoke_v4/`

## What to put in each folder

**One file:** the version's **Output** tab → download
`ror-output_<runs>_<time>.zip` and extract it into the folder. The notebook's
last cell writes it; it contains:

- `session_log.txt`: the output of every command in the session (tests, plan,
  training, inference), with progress bars thinned out;
- `requirements.lock`: the exact library versions of that session;
- `runs_smoke/` or `runs/`: the registry, `results.jsonl` and one folder per
  experiment, without the adapter weights (those stay in the Kaggle output,
  where the next session restores them from).

If the zip is missing (the notebook stopped before its last cell), download the
version's **Logs** file instead and put it in the folder.

```
Results/
  2026-09-25_smoke_v4/
    session_log.txt
    requirements.lock
    runs_smoke/
      results.jsonl
      <exp_id>/  run.log  status.json  result.json  predictions.jsonl
                 train_stats.json  adapter/adapter_config.json
```
