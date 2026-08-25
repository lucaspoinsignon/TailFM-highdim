```bash
python -c "
import numpy as np, pandas as pd
df = pd.read_csv('data/returns_clean.csv', index_col=0, parse_dates=True)
sel = np.random.default_rng(0).choice(df.columns, 30, replace=False)
out = df.loc[:, df.columns.isin(sel)]          
out.to_csv('data/returns_30.csv')
print(out.shape, list(out.columns))
"

python check.py --data data/returns_30.csv
python fit_returns.py --data data/returns_30.csv --outdir runs/f30

```
