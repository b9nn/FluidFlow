"""
BM extended — Result B: runtime, adaptation cost vs inference cost.

Closes the loop with the previous paper's motivation: a surrogate must be
accurate AND practical. We separate, per method:
  - PRETRAINING (paid once, not per task): TabPFN's synthetic-prior pretrain
    (amortized, ~0 here) vs TransRF's BCM source RF fit. Reported once.
  - ADAPTATION (per task, given N target rows): TabPFN in-context fit vs
    TransRF fitting target/residual/augmented RFs + ensemble weights.
  - INFERENCE (per task): predicting a held-out test pool.

Timed at a fixed N, averaged over repeats, for all 5 outputs.
Note: TabPFN here is the CLOUD client, so its times include network round-trip
(a practical-cost view, flagged in the figure).

Output: figs/bm_ext_runtime.png  + console table
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tabpfn_client import TabPFNRegressor
try:
    from tabpfn import TabPFNRegressor as TabPFNLocal
    _HAVE_LOCAL = True
except Exception:
    _HAVE_LOCAL = False

import BM_Extended_TransferRF as T   # reuse FEATURES/TARGETS/get_model_params/load

script_dir = os.path.dirname(os.path.abspath(__file__))
FEATURES, TARGETS = T.FEATURES, T.TARGETS
N_FIT = 200
N_TEST = 1000
REPEATS = 3


def time_transrf(df_bcm, df_bm):
    # Pretraining: source RF fit (once)
    t0 = time.perf_counter()
    sources = {t: T.train_source(df_bcm, t) for t in TARGETS}
    pretrain = time.perf_counter() - t0

    adapt, infer = [], []
    for r in range(REPEATS):
        tr = df_bm.sample(n=N_FIT, random_state=42 + r)
        te = df_bm.drop(tr.index).sample(n=N_TEST, random_state=42 + r)
        Xtr, Xte = tr[FEATURES], te[FEATURES]
        params = T.get_model_params(N_FIT)
        ta = ti = 0.0
        for tgt in TARGETS:
            rf_src, sc_src = sources[tgt]
            ytr = tr[tgt].values
            t0 = time.perf_counter()
            sc = StandardScaler(); Xtr_sc = sc.fit_transform(Xtr)
            ps_tr = rf_src.predict(sc_src.transform(Xtr))
            rf_t = RandomForestRegressor(n_jobs=-1, random_state=r, **params).fit(Xtr_sc, ytr)
            rf_r = RandomForestRegressor(n_jobs=-1, random_state=r, **params).fit(Xtr.values, ytr - ps_tr)
            Xtra = np.column_stack([Xtr.values, ps_tr])
            rf_a = RandomForestRegressor(n_jobs=-1, random_state=r, **params).fit(Xtra, ytr)
            stack = np.column_stack([rf_t.predict(Xtr_sc), ps_tr + rf_r.predict(Xtr.values),
                                     rf_a.predict(Xtra)])
            LinearRegression(positive=True, fit_intercept=False).fit(stack, ytr)
            ta += time.perf_counter() - t0
            t0 = time.perf_counter()
            ps_te = rf_src.predict(sc_src.transform(Xte))
            _ = rf_t.predict(sc.transform(Xte)); _ = ps_te + rf_r.predict(Xte.values)
            _ = rf_a.predict(np.column_stack([Xte.values, ps_te]))
            ti += time.perf_counter() - t0
        adapt.append(ta); infer.append(ti)
    return pretrain, np.mean(adapt), np.mean(infer)


def time_tabpfn(df_bm, RegClass):
    adapt, infer = [], []
    for r in range(REPEATS):
        tr = df_bm.sample(n=N_FIT, random_state=42 + r)
        te = df_bm.drop(tr.index).sample(n=N_TEST, random_state=42 + r)
        sc = StandardScaler(); Xtr = sc.fit_transform(tr[FEATURES].values)
        Xte = sc.transform(te[FEATURES].values)
        ta = ti = 0.0
        for tgt in TARGETS:
            sy = StandardScaler(); y = sy.fit_transform(tr[[tgt]].values).ravel()
            reg = RegClass(random_state=r)
            t0 = time.perf_counter(); reg.fit(Xtr, y); ta += time.perf_counter() - t0
            t0 = time.perf_counter(); _ = reg.predict(Xte); ti += time.perf_counter() - t0
        adapt.append(ta); infer.append(ti)
    return 0.0, np.mean(adapt), np.mean(infer)   # pretrain amortized (~0 per use)


def main():
    print("=" * 70)
    print(f"BM EXTENDED Result B — runtime (N={N_FIT} fit, {N_TEST} test, {REPEATS} reps, 5 outputs)")
    print("=" * 70)
    df_bcm = pd.read_csv(os.path.join(script_dir, '..', 'TBCM', 'dataset_BCM.csv'),
                         index_col=0).rename(columns={'Ps': 'PS'})
    df_bm = T.load_bm()

    pt_tr, ad_tr, in_tr = time_transrf(df_bcm, df_bm)
    print(f"  TransRF       pretrain(source RF)={pt_tr:6.2f}s  adapt={ad_tr:6.2f}s  infer={in_tr:7.3f}s")
    _, ad_tbc, in_tbc = time_tabpfn(df_bm, TabPFNRegressor)
    print(f"  TabPFN-cloud  pretrain(amortized)=  0.00s  adapt={ad_tbc:6.2f}s  infer={in_tbc:7.3f}s  (incl network)")
    methods = ['TransRF', 'TabPFN\n(cloud)']
    adapt = [ad_tr, ad_tbc]
    infer = [in_tr, in_tbc]
    if _HAVE_LOCAL:
        try:
            _, ad_tbl, in_tbl = time_tabpfn(df_bm, TabPFNLocal)
            print(f"  TabPFN-local  pretrain(amortized)=  0.00s  adapt={ad_tbl:6.2f}s  infer={in_tbl:7.3f}s  (no network)")
            methods.append('TabPFN\n(local)'); adapt.append(ad_tbl); infer.append(in_tbl)
        except Exception as e:
            print(f"  TabPFN-local  unavailable: {type(e).__name__}: {e}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(methods))
    ax.bar(x, adapt, 0.5, label='Adaptation (per task)', color='#2563eb')
    ax.bar(x, infer, 0.5, bottom=adapt, label=f'Inference ({N_TEST} preds)', color='#d97706')
    for i, (a, b) in enumerate(zip(adapt, infer)):
        ax.text(i, a / 2, f'{a:.2f}s', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, a + b / 2, f'{b:.2f}s', ha='center', va='center', color='white', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel('wall-clock seconds (5 outputs)')
    ax.set_title(f'BCM→BM runtime per task @ N={N_FIT}  —  adaptation vs inference\n'
                 f'(TransRF source-RF pretrain {pt_tr:.1f}s paid once; TabPFN cloud incl. network)',
                 fontweight='bold')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(script_dir, 'figs', 'bm_ext_runtime.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == '__main__':
    main()
