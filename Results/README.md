# Results — Kaggle run outputs, one folder per run

Drop each Kaggle run here in its own folder. Claude reads them from this folder.
The contents are gitignored (the repo is public and adapters are large); only
this README is tracked.

## Naming

`YYYY-MM-DD_<suite>_v<kaggle version>`, e.g. `2026-09-25_smoke_v2/`

## What to put in each folder

1. **The output:** on the Kaggle notebook page → the version → **Output** →
   download all, and extract it here. It contains `runs_smoke/` or `runs/`
   (registry, `results.jsonl`, per-experiment folders) and `requirements.lock`.
2. **The log:** the version's **Logs** download (e.g. `ror-c.log`), same folder.
   It shows the full console output if something failed.

Adapter weights (`adapter/*.safetensors`) can be left out if space matters. The
metrics, predictions and logs are what is needed to read a run.

```
Results/
  2026-09-25_smoke_v2/
    ror-c.log
    requirements.lock
    runs_smoke/
      results.jsonl
      <exp_id>/  run.log  status.json  result.json  predictions.jsonl
                 train_stats.json  adapter/
```
