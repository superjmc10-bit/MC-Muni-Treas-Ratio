# Your ratio history (optional)

Drop a CSV here and point `valuation.history_csv` at it to replace the
approximate history statistics in `config/default.yaml`:

```bash
python -m muni_mc run --set valuation.history_csv=data/my_mmd_history.csv
```

Format (see `history_template.csv`):

| column | meaning |
|---|---|
| `date` | any date pandas can parse (daily or weekly) |
| `ratio_<tenor>` | MMD AAA / UST ratio for that tenor, in decimals (0.64) or percent (64) |
| `ust_<tenor>` | optional: UST yield in %; used to calibrate the rate-level term |

Tenors don't need to match the model's tenors; the model interpolates them.
Only rows inside `valuation.lookback_years` before `meta.as_of` are used.
No data ships with the repo. The template has headers only, so nothing here is
mistaken for real history.
