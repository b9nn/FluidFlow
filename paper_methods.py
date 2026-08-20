"""
Shared method implementations for the FluidFlow low-N surrogate paper (Draft 5).

Defines the seven configurations evaluated in the paper under ONE protocol:

  baseline (fit on target alone) : PR, RF, NN
  optimized transfer            : PR, RF, NN   (identical procedure per family)
  no source                     : TabPFN

plus `source alone` (BCM source applied to the target with no adaptation).

The optimized-transfer procedure is the SAME for all three regressor families,
exactly as described in the paper's Methods:

    source model  (fit once on the BCM source domain)
      + sub-model A : family fit on the N target rows                (= baseline)
      + sub-model B : family fit on the residual Y - source_pred     (residual correction)
      + sub-model C : family fit on [X, source_pred] -> Y            (feature augmentation)
    blended by non-negative weights fit on OUT-OF-FOLD predictions.

Out-of-fold (K = min(5, N)) stacking is used for the blend weights so that
sub-model A cannot win weight simply by memorising the N training rows. See
docs/DECISIONS.md 2026-08-20.

Per CLAUDE.md hard convention #1, every domain fits its own input scaler and its
own per-output scalers; no scaler is ever shared across domains.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn

DEVICE = torch.device("cpu")


# ============================================================================
# metrics
# ============================================================================
def nrmse(y_true, y_pred):
    """Range-normalised RMSE (paper Eq. 1). Range is taken over the test set."""
    rng = float(np.max(y_true) - np.min(y_true))
    if rng <= 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / rng)


def r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


# ============================================================================
# adaptive RF complexity  (CLAUDE.md hard convention #4)
# ============================================================================
def get_model_params(n_samples):
    """Scale RF capacity with sample count.

    Tree count does not drive overfitting in a bagged ensemble, so it is held
    high throughout; capacity is controlled by depth and leaf size. Recalibrated
    for the low-N regime (the previous schedule forced min_samples_leaf=10 at
    N=10, which collapses every tree to a stump). See DECISIONS 2026-08-20.
    """
    if n_samples < 50:
        return dict(n_estimators=300, max_depth=3, min_samples_leaf=1, min_samples_split=2)
    if n_samples < 250:
        return dict(n_estimators=300, max_depth=6, min_samples_leaf=2, min_samples_split=4)
    if n_samples < 1000:
        return dict(n_estimators=300, max_depth=10, min_samples_leaf=2, min_samples_split=4)
    return dict(n_estimators=300, max_depth=None, min_samples_leaf=1, min_samples_split=2)


# ============================================================================
# regressor families.  Each .fit(X, Y, n_hint) -> object exposing .predict(X)
# Every family standardises its own inputs and its own per-output targets.
# ============================================================================
class _Fitted:
    def __init__(self, predict_fn):
        self._predict = predict_fn

    def predict(self, X):
        return self._predict(np.asarray(X, dtype=float))


class PolyFamily:
    """Degree-3 polynomial features + ridge with per-target alpha chosen by
    leave-one-out generalised cross-validation."""

    name = "PR"
    DEGREE = 3
    ALPHAS = np.logspace(-3, 4, 15)

    def fit(self, X, Y, n_hint=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(Y)
        poly = PolynomialFeatures(degree=self.DEGREE, include_bias=False)
        P = poly.fit_transform(sx.transform(X))
        sp = StandardScaler().fit(P)
        model = RidgeCV(alphas=self.ALPHAS, alpha_per_target=True)
        model.fit(sp.transform(P), sy.transform(Y))

        def _predict(Xq):
            Pq = sp.transform(poly.transform(sx.transform(Xq)))
            return sy.inverse_transform(model.predict(Pq))

        return _Fitted(_predict)


class RFFamily:
    """Random forest with sample-size-adaptive depth/leaf size."""

    name = "RF"

    def fit(self, X, Y, n_hint=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = len(X) if n_hint is None else n_hint
        sx = StandardScaler().fit(X)
        model = RandomForestRegressor(random_state=42, n_jobs=-1, **get_model_params(n))
        model.fit(sx.transform(X), Y)

        def _predict(Xq):
            out = model.predict(sx.transform(Xq))
            return out.reshape(len(Xq), -1)

        return _Fitted(_predict)


class NNFamily:
    """Feed-forward MLP, 2 hidden layers of 64 ReLU units, shared trunk with one
    linear output per target. Full-batch Adam on standardised inputs/outputs."""

    name = "NN"
    HIDDEN = 64
    EPOCHS = 800
    LR = 3e-3
    WEIGHT_DECAY = 1e-4

    def fit(self, X, Y, n_hint=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(Y)
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Linear(X.shape[1], self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, Y.shape[1]),
        ).to(DEVICE)
        Xt = torch.FloatTensor(sx.transform(X)).to(DEVICE)
        Yt = torch.FloatTensor(sy.transform(Y)).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.LR,
                               weight_decay=self.WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        net.train()
        for _ in range(self.EPOCHS):
            opt.zero_grad()
            loss_fn(net(Xt), Yt).backward()
            opt.step()
        net.eval()

        def _predict(Xq):
            with torch.no_grad():
                p = net(torch.FloatTensor(sx.transform(Xq)).to(DEVICE)).cpu().numpy()
            return sy.inverse_transform(p)

        return _Fitted(_predict)


class NNSourceFamily(NNFamily):
    """NN family variant used for the large BCM source fit: mini-batch training
    on a subsample, since full-batch over 360k rows is neither needed nor fast."""

    EPOCHS = 300
    LR = 1e-3
    BATCH = 256
    SUBSAMPLE = 20000

    def fit(self, X, Y, n_hint=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        rng = np.random.RandomState(42)
        if len(X) > self.SUBSAMPLE:
            idx = rng.choice(len(X), self.SUBSAMPLE, replace=False)
            X, Y = X[idx], Y[idx]
        sx = StandardScaler().fit(X)
        sy = StandardScaler().fit(Y)
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Linear(X.shape[1], self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, self.HIDDEN), nn.ReLU(),
            nn.Linear(self.HIDDEN, Y.shape[1]),
        ).to(DEVICE)
        Xt = torch.FloatTensor(sx.transform(X)).to(DEVICE)
        Yt = torch.FloatTensor(sy.transform(Y)).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=self.LR,
                               weight_decay=self.WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(Xt, Yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.BATCH, shuffle=True)
        net.train()
        for _ in range(self.EPOCHS):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
        net.eval()

        def _predict(Xq):
            with torch.no_grad():
                p = net(torch.FloatTensor(sx.transform(Xq)).to(DEVICE)).cpu().numpy()
            return sy.inverse_transform(p)

        return _Fitted(_predict)


FAMILIES = {"PR": PolyFamily(), "RF": RFFamily(), "NN": NNFamily()}
SOURCE_FAMILIES = {"PR": PolyFamily(), "RF": RFFamily(), "NN": NNSourceFamily()}


# ============================================================================
# the optimized transfer procedure (identical for every family)
# ============================================================================
def _submodel_predictions(family, X_tr, Y_tr, src_tr, X_ev, src_ev, n_hint):
    """Fit sub-models A/B/C on (X_tr, Y_tr) and predict at X_ev.

    A: target only.  B: residual correction.  C: feature augmentation.
    Returns (pA, pB, pC), each (len(X_ev), n_outputs).
    """
    mA = family.fit(X_tr, Y_tr, n_hint)
    pA = mA.predict(X_ev)

    mB = family.fit(X_tr, Y_tr - src_tr, n_hint)
    pB = src_ev + mB.predict(X_ev)

    mC = family.fit(np.hstack([X_tr, src_tr]), Y_tr, n_hint)
    pC = mC.predict(np.hstack([X_ev, src_ev]))

    return pA, pB, pC


def optimized_transfer(family, X_tr, Y_tr, X_te, source_model,
                       seed=42, return_baseline=True):
    """Run the paper's optimized transfer procedure for one regressor family.

    Returns (transfer_pred, baseline_pred, weights) where `weights` is
    (n_outputs, 3), rows summing to 1, columns = [target-only, residual, augmented].
    """
    X_tr = np.asarray(X_tr, dtype=float)
    Y_tr = np.asarray(Y_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    n, n_out = len(X_tr), Y_tr.shape[1]

    src_tr = source_model.predict(X_tr)
    src_te = source_model.predict(X_te)

    # ---- out-of-fold sub-model predictions, used only to fit blend weights ----
    K = int(min(5, n))
    oof = np.full((n, n_out, 3), np.nan)
    if K >= 2:
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        for tr_idx, ev_idx in kf.split(X_tr):
            if len(tr_idx) < 2:
                continue
            pA, pB, pC = _submodel_predictions(
                family, X_tr[tr_idx], Y_tr[tr_idx], src_tr[tr_idx],
                X_tr[ev_idx], src_tr[ev_idx], n_hint=n)
            oof[ev_idx, :, 0] = pA
            oof[ev_idx, :, 1] = pB
            oof[ev_idx, :, 2] = pC

    # ---- refit sub-models on all N rows, predict the test pool ----
    pA, pB, pC = _submodel_predictions(
        family, X_tr, Y_tr, src_tr, X_te, src_te, n_hint=n)
    stack_te = np.stack([pA, pB, pC], axis=2)          # (n_test, n_out, 3)

    # ---- non-negative blend weights per output, fit on the OOF stack ----
    weights = np.zeros((n_out, 3))
    for j in range(n_out):
        Z = oof[:, j, :]
        ok = np.all(np.isfinite(Z), axis=1)
        if ok.sum() >= 3:
            reg = LinearRegression(positive=True, fit_intercept=False)
            reg.fit(Z[ok], Y_tr[ok, j])
            w = np.clip(reg.coef_, 0, None)
        else:
            w = np.zeros(3)
        if w.sum() <= 0:
            w = np.ones(3) / 3.0
        weights[j] = w / w.sum()

    transfer_pred = np.einsum("tof,of->to", stack_te, weights)
    baseline_pred = pA if return_baseline else None
    return transfer_pred, baseline_pred, weights


# ============================================================================
# TabPFN (no source)
# ============================================================================
_TABPFN_MAX_TRAIN = 1000


def tabpfn_predict(X_tr, Y_tr, X_te, seed=42, regressor_cls=None):
    """One independent single-output TabPFN head per target, each with its own
    input and output standardisation. No training, no source model."""
    if regressor_cls is None:
        from tabpfn_client import TabPFNRegressor as regressor_cls_  # noqa: N813
        regressor_cls = regressor_cls_
    X_tr = np.asarray(X_tr, dtype=float)[:_TABPFN_MAX_TRAIN]
    Y_tr = np.asarray(Y_tr, dtype=float)[:_TABPFN_MAX_TRAIN]
    X_te = np.asarray(X_te, dtype=float)
    out = np.zeros((len(X_te), Y_tr.shape[1]))
    for j in range(Y_tr.shape[1]):
        sx = StandardScaler().fit(X_tr)
        sy = StandardScaler().fit(Y_tr[:, [j]])
        reg = regressor_cls(random_state=seed)
        reg.fit(sx.transform(X_tr), sy.transform(Y_tr[:, [j]]).ravel())
        p = reg.predict(sx.transform(X_te)).reshape(-1, 1)
        out[:, j] = sy.inverse_transform(p).ravel()
    return out


def enable_tabpfn():
    """Authenticate the tabpfn cloud client from the cached token / env var."""
    from tabpfn_client.service_wrapper import UserAuthenticationClient
    ok, _ = UserAuthenticationClient.try_reuse_existing_token()
    return bool(ok)
