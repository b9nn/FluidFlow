"""Recompute ONLY the NN-transfer column (now full-network fine-tune) and merge
into results/bm_five_method.json, then regenerate the four-method figure. Local
only, no TabPFN cloud calls. Reuses BM_FiveMethod's protocol, seeds, and source."""
import json, os
import numpy as np, pandas as pd
import BM_FiveMethod as M

script_dir = M.script_dir
df_src = pd.read_parquet(os.path.join(script_dir, '..', 'data_binary.parquet'))
if 'Ps' in df_src.columns and 'PS' not in df_src.columns:
    df_src = df_src.rename(columns={'Ps': 'PS'})
df_src = df_src.dropna(subset=M.OUTPUTS)[M.FEATURES + M.OUTPUTS]
df_bm = pd.read_csv(os.path.join(script_dir, 'dataset_BM_extended.csv'))
if 'Ps' in df_bm.columns and 'PS' not in df_bm.columns:
    df_bm = df_bm.rename(columns={'Ps': 'PS'})
df_bm = df_bm.dropna(subset=M.OUTPUTS)[M.FEATURES + M.OUTPUTS]

print('Rebuilding source models (for NN)...')
src = M.SourceModels(df_src)

res_path = os.path.join(script_dir, 'results', 'bm_five_method.json')
results = json.load(open(res_path))

for n in M.N_GRID:
    r2 = {o: [] for o in M.OUTPUTS}
    nr = {o: [] for o in M.OUTPUTS}
    for r in range(M.N_RUNS):
        rs = M.RANDOM_STATE + r
        tr = df_bm.sample(n=n, random_state=rs)
        pool = df_bm.drop(tr.index)
        te = pool.sample(n=min(M.TEST_POOL_SIZE, len(pool)), random_state=rs)
        Xtr, Ytr = tr[M.FEATURES].values, tr[M.OUTPUTS].values
        Xte, Yte = te[M.FEATURES].values, te[M.OUTPUTS].values
        pred = M.predict_nn(src, Xtr, Ytr, Xte)
        for j, o in enumerate(M.OUTPUTS):
            r2[o].append(M.r2_score(Yte[:, j], pred[:, j]))
            nr[o].append(M.nrmse(Yte[:, j], pred[:, j]))
    results['r2']['NN'][str(n)] = {o: {'mean': float(np.mean(r2[o])), 'std': float(np.std(r2[o]))} for o in M.OUTPUTS}
    results['nrmse']['NN'][str(n)] = {o: {'mean': float(np.nanmean(nr[o])), 'std': float(np.nanstd(nr[o]))} for o in M.OUTPUTS}
    print(f"  N={n:>4}  NN " + ' '.join(f"{o}={np.mean(r2[o]):+.2f}" for o in M.OUTPUTS))

json.dump(results, open(res_path, 'w'), indent=2)
M._plot(results, ['PR', 'RF', 'NN', 'TabPFN'])
print('merged + re-plotted')
