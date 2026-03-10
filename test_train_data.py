import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('/Users/ishraklatif/Documents/financial_data/Financial-data')

with open(ROOT / 'config/features_locked.json') as f:
    features = json.load(f)['features']

train = pd.read_parquet(ROOT / 'data/features/panel_train_scaled.parquet')
val   = pd.read_parquet(ROOT / 'data/features/panel_val_scaled.parquet')

X_tr = train[features].astype('float32').values
y_tr = train['y_ret_21d'].astype('float32').values
X_va = val[features].astype('float32').values
y_va = val['y_ret_21d'].astype('float32').values
dates_va = val['date'].astype(str).values

train_mask = ~np.isnan(y_tr)
val_mask   = ~np.isnan(y_va)

X_tr_m = X_tr[train_mask]
y_tr_m = y_tr[train_mask]
X_va_m = X_va[val_mask]
y_va_m = y_va[val_mask]
dates_va_m = dates_va[val_mask]

dtrain = lgb.Dataset(X_tr_m, label=y_tr_m, free_raw_data=False)

def compute_ic(pred, y, dates, min_names=5):
    df = pd.DataFrame({'pred': pred, 'y': y, 'date': dates}).dropna()
    ics = []
    for date, grp in df.groupby('date'):
        if len(grp) < min_names:
            continue
        if grp['pred'].nunique() < 2 or grp['y'].nunique() < 2:
            continue
        ic, _ = spearmanr(grp['pred'], grp['y'])
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(ics)), float(np.std(ics)), float(np.mean(ics > 0))


def train_with_ic_stopping(params, dtrain, X_va, y_va, dates_va,
                            num_rounds=2000, patience=100, eval_every=5,
                            min_ic_dates=500):
    model = None
    best_ic = -np.inf
    best_iter = 0
    no_improve = 0
    best_model = None

    for i in range(1, num_rounds + 1):
        if model is None:
            model = lgb.train(params, dtrain, num_boost_round=1,
                              callbacks=[lgb.log_evaluation(period=0)])
        else:
            model = lgb.train(params, dtrain, num_boost_round=1,
                              init_model=model,
                              callbacks=[lgb.log_evaluation(period=0)])

        if i % eval_every == 0:
            pred_va = model.predict(X_va)

            df = pd.DataFrame({'pred': pred_va, 'y': y_va, 'date': dates_va}).dropna()
            n_valid_dates = sum(
                1 for _, grp in df.groupby('date')
                if len(grp) >= 5
                and grp['pred'].nunique() >= 2
                and grp['y'].nunique() >= 2
            )

            if n_valid_dates < min_ic_dates:
                continue

            ic_mean, ic_std, ic_pos = compute_ic(pred_va, y_va, dates_va)

            if ic_mean > best_ic:
                best_ic = ic_mean
                best_iter = i
                no_improve = 0
                best_model = model
            else:
                no_improve += eval_every

            if no_improve >= patience:
                break

    return best_model, best_iter, best_ic


configs5 = [
    ('nobag_col04',   15, 0.01, 200, 0.0, 10.0, 0.4, 1.0, 0),
    ('nobag_col05',   15, 0.01, 200, 0.0, 10.0, 0.5, 1.0, 0),
    ('ss05_col04',    15, 0.01, 200, 0.0, 10.0, 0.4, 0.5, 5),
    ('best_base',     15, 0.01, 200, 0.1, 10.0, 0.4, 0.7, 5),
    ('nobag_nl10',    10, 0.01, 200, 0.0, 10.0, 0.4, 1.0, 0),
    ('nobag_mcs300',  15, 0.01, 300, 0.0, 10.0, 0.4, 1.0, 0),
]

print(f"\n{'Config':<20} {'BestIter':>8} {'BestIC':>8} {'ICIR':>7} {'IC+%':>6} {'n_dates':>8}")
print("-" * 65)

for name, nl, lr, mcs, ra, rl, cf, ss, bf in configs5:
    params = {
        'objective':         'regression',
        'metric':            'None',
        'num_leaves':        nl,
        'learning_rate':     lr,
        'feature_fraction':  cf,
        'bagging_fraction':  ss,
        'bagging_freq':      bf,
        'min_child_samples': mcs,
        'reg_alpha':         ra,
        'reg_lambda':        rl,
        'verbose':           -1,
        'n_jobs':            -1,
    }

    best_model, best_iter, best_ic = train_with_ic_stopping(
        params, dtrain, X_va_m, y_va_m, dates_va_m,
        num_rounds=2000, patience=150, eval_every=5,
        min_ic_dates=500
    )

    if best_model is None:
        print(f"  {name:<20} {'N/A':>8} {'N/A':>8}")
        continue

    pred_va = best_model.predict(X_va_m)
    ic_mean, ic_std, ic_pos = compute_ic(pred_va, y_va_m, dates_va_m)
    icir = ic_mean / ic_std if (ic_std and ic_std > 0) else 0

    df = pd.DataFrame({'pred': pred_va, 'y': y_va_m, 'date': dates_va_m}).dropna()
    n_dates = sum(1 for _, grp in df.groupby('date')
                  if len(grp) >= 5 and grp['pred'].nunique() >= 2)

    print(f"  {name:<20} {best_iter:>8} {ic_mean:>+8.4f} {icir:>7.3f} {ic_pos:>5.1%} {n_dates:>8}")