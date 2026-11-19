#!/usr/bin/env python3
"""A6P1 - Explainable decision automation for rooftop-solar interconnection approvals.

Runs every experiment of the paper end to end:
  preprocessing (median imputation, train-fitted winsorisation, screen features),
  proposed process-aligned two-stage P-spline GAM (PS-GAM) with selective automation,
  baselines (manual screening rules, multinomial logistic regression, LightGBM + TreeSHAP),
  5-fold stratified CV (mean +- std, paired t-tests), temporal drift split (train 2025 -> test 2026,
  bootstrap CIs, McNemar), selective-automation curves, explanation faithfulness (deletion AOPC,
  distractor share, cross-fold stability), counterfactual recourse, ablation and smoothing sensitivity.
Outputs: ../results/*.csv, ../results/metrics.json, ../figures/*.png (300 dpi) and ../results/*.jpg.
"""
import os, json, time, itertools, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "A6P1_solar_interconnection_apps.csv")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 2026
CLASSES = ["fast_track_approve", "supplemental_review", "reject_or_rework"]  # 0=A, 1=S, 2=R
SHORT = ["Approve", "Review", "Reject/rework"]
T0 = time.time()

# ----------------------------------------------------------------------------- features
SPLINE = ["pv_system_kw_dc", "transformer_existing_pv_kw", "feeder_hosting_capacity_kw", "feeder_der_penetration",
          "distance_to_substation_km", "installer_quality_score", "document_completeness", "roof_age_years",
          "remaining_hc_kw", "xfmr_util", "service_ratio"]
LINEAR = ["battery_kwh", "service_capacity_kva", "service_transformer_kva", "prior_applications_at_site",
          "battery_attached"]
CATS = {"customer_class": ["small_commercial", "multi_family"],
        "inverter_certification": ["ul1741sa_legacy", "uncertified"],
        "export_mode": ["limited_export", "non_export"]}
INTER = "battery_x_managed_export"
SCREEN = ["remaining_hc_kw", "xfmr_util", "service_ratio"]
RAW_NUM = [c for c in SPLINE + LINEAR if c not in SCREEN]
LEAK = ["processing_days"]  # post-decision column: never used as a feature
GROUPS = SPLINE + LINEAR + list(CATS) + [INTER]
NICE = {"pv_system_kw_dc": "PV size", "transformer_existing_pv_kw": "Xfmr existing PV",
        "feeder_hosting_capacity_kw": "Hosting capacity", "feeder_der_penetration": "DER penetration",
        "distance_to_substation_km": "Distance to sub.", "installer_quality_score": "Installer score",
        "document_completeness": "Doc. completeness", "roof_age_years": "Roof age (distractor)",
        "remaining_hc_kw": "Remaining HC", "xfmr_util": "Xfmr utilisation", "service_ratio": "Service ratio",
        "battery_kwh": "Battery kWh", "service_capacity_kva": "Service kVA", "service_transformer_kva": "Xfmr kVA",
        "prior_applications_at_site": "Prior apps", "battery_attached": "Battery", "customer_class": "Customer class",
        "inverter_certification": "Inverter cert.", "export_mode": "Export mode", INTER: "Battery x managed export"}


def load():
    df = pd.read_csv(DATA, parse_dates=["submitted_at"])
    df["y"] = df["decision"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return df


class Prep:
    """Median imputation + winsorisation (0.5/99.5 pct) fitted on the training fold, then screen features."""
    def __init__(self, winsor=True):
        self.winsor = winsor

    def _screens(self, d):
        d["remaining_hc_kw"] = d["feeder_hosting_capacity_kw"] * (1 - d["feeder_der_penetration"])
        exp = (d["export_mode"] != "non_export").astype(float)
        d["xfmr_util"] = (d["transformer_existing_pv_kw"] + d["pv_system_kw_dc"] * exp) / d["service_transformer_kva"]
        d["service_ratio"] = d["pv_system_kw_dc"] / d["service_capacity_kva"]
        return d

    def fit(self, df):
        d = df.copy()
        self.med = {c: d[c].median() for c in RAW_NUM}
        for c in RAW_NUM: d[c] = d[c].fillna(self.med[c])
        self.lo = {c: d[c].quantile(0.005) for c in RAW_NUM}; self.hi = {c: d[c].quantile(0.995) for c in RAW_NUM}
        if self.winsor:
            for c in RAW_NUM: d[c] = d[c].clip(self.lo[c], self.hi[c])
        d = self._screens(d)
        self.lo.update({c: d[c].quantile(0.005) for c in SCREEN}); self.hi.update({c: d[c].quantile(0.995) for c in SCREEN})
        return self

    def transform(self, df):
        d = df.copy()
        for c in RAW_NUM: d[c] = d[c].fillna(self.med[c])
        if self.winsor:
            for c in RAW_NUM: d[c] = d[c].clip(self.lo[c], self.hi[c])
        d = self._screens(d)
        if self.winsor:
            for c in SCREEN: d[c] = d[c].clip(self.lo[c], self.hi[c])
        X = d[SPLINE + LINEAR].astype(float).copy()
        for c, levs in CATS.items():
            for l in levs: X[f"{c}={l}"] = (d[c] == l).astype(float)
        X[INTER] = (d["battery_attached"] * (d["export_mode"] != "full_export")).astype(float)
        return X


def group_cols(cols):
    """Map each feature group to its column indices in the tabular matrix."""
    g = {}
    for i, c in enumerate(cols):
        key = c.split("=")[0]
        g.setdefault(key, []).append(i)
    return [g[k] for k in GROUPS]


# ----------------------------------------------------------------------------- proposed: P-spline GAM
class PSplineGAM:
    """Binary logistic GAM: logit p = b0 + sum_j f_j(x_j) + linear + categorical terms.
    f_j = cubic B-spline basis (n_seg equal segments). Each spline term carries a double penalty
    lam_j1 ||D2 beta_j||^2 + lam_j2 ||N_j beta_j||^2 (wiggliness + null-space shrinkage, so a useless term can be
    shrunk to zero). With auto=True the smoothing parameters are estimated term by term with the generalised
    Fellner-Schall update; with auto=False a single fixed lam is used for every wiggliness penalty.
    Coefficients are fitted by penalised IRLS (PIRLS) with step halving."""
    def __init__(self, lam=1.0, n_seg=8, deg=3, linear_only=(), lam_lin=1e-2, lam_cat=0.3, auto=True,
                 lam_null=None, max_outer=60):
        self.lam, self.n_seg, self.deg = lam, n_seg, deg
        self.linear_only = set(linear_only); self.lam_lin, self.lam_cat = lam_lin, lam_cat
        self.auto, self.max_outer = auto, max_outer
        self.lam_null = lam if lam_null is None else lam_null

    def _setup(self, X):
        self.cols = list(X.columns); self.terms = []
        for c in self.cols:
            base = c.split("=")[0]
            if c in SPLINE and c not in self.linear_only:
                lo, hi = float(X[c].min()), float(X[c].max())
                step = (hi - lo) / self.n_seg
                t = np.concatenate([lo - step * np.arange(self.deg, 0, -1), np.linspace(lo, hi, self.n_seg + 1),
                                    hi + step * np.arange(1, self.deg + 1)])
                self.terms.append(dict(col=c, group=base, kind="spline", t=t, lo=lo, hi=hi))
            else:
                self.terms.append(dict(col=c, group=base, kind="lin", mu=float(X[c].mean()), sd=float(X[c].std() + 1e-9)))

    def _basis(self, X, center=True):
        blocks, sizes = [], []
        for tm in self.terms:
            x = X[tm["col"]].to_numpy(float)
            if tm["kind"] == "spline":
                xc = np.clip(x, tm["lo"], tm["hi"])
                B = BSpline.design_matrix(xc, tm["t"], self.deg).toarray()
                if center:
                    B = B - tm["bmean"] if "bmean" in tm else B
            else:
                B = ((x - tm["mu"]) / tm["sd"])[:, None]
            blocks.append(B); sizes.append(B.shape[1])
        return np.hstack(blocks), sizes

    def _pirls(self, Z, y, w, P, beta):
        def pdev(b):
            e = np.clip(Z @ b, -30, 30); m_ = 1 / (1 + np.exp(-e))
            return -2 * np.sum(w * (y * np.log(m_ + 1e-12) + (1 - y) * np.log(1 - m_ + 1e-12))) + b @ P @ b
        dev_old = pdev(beta)
        for it in range(100):
            eta = np.clip(Z @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
            v = np.clip(mu * (1 - mu), 1e-6, None); W = v * w
            zz = eta + (y - mu) / v
            A = Z.T @ (W[:, None] * Z) + P
            step = np.linalg.solve(A, Z.T @ (W * zz)) - beta
            a = 1.0
            for _ in range(30):  # step halving keeps the penalised deviance monotone
                dev = pdev(beta + a * step)
                if dev <= dev_old + 1e-9: break
                a *= 0.5
            beta = beta + a * step
            if abs(dev_old - dev) < 1e-8 * (abs(dev) + 1): break
            dev_old = dev
        eta = np.clip(Z @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-6, None) * w
        ZtWZ = Z.T @ (W[:, None] * Z)
        return beta, ZtWZ + P, ZtWZ, it + 1

    def fit(self, X, y, w=None):
        self._setup(X)
        for tm in self.terms:
            if tm["kind"] == "spline":
                xc = np.clip(X[tm["col"]].to_numpy(float), tm["lo"], tm["hi"])
                tm["bmean"] = BSpline.design_matrix(xc, tm["t"], self.deg).toarray().mean(0)
        B, sizes = self._basis(X)
        self.sizes = sizes
        n, p = B.shape
        Z = np.hstack([np.ones((n, 1)), B])
        P0 = np.zeros((p + 1, p + 1)); k = 1; pens = []   # pens: (slice, S, rank) for each smoothing parameter
        for tm, s in zip(self.terms, sizes):
            if tm["kind"] == "spline":
                D = np.diff(np.eye(s), n=2, axis=0); S1 = D.T @ D
                ev, U = np.linalg.eigh(S1); U0 = U[:, ev < 1e-8]; S2 = U0 @ U0.T   # null space (constant + linear)
                tm["pen"] = [len(pens), len(pens) + 1]
                pens.append((slice(k, k + s), S1, s - 2)); pens.append((slice(k, k + s), S2, U0.shape[1]))
                P0[k:k + s, k:k + s] += 1e-6 * np.eye(s)
            else:
                P0[k, k] = self.lam_cat if "=" in tm["col"] or tm["col"] == INTER else self.lam_lin
            k += s
        lams = np.array([self.lam if i % 2 == 0 else self.lam_null for i in range(len(pens))], float)
        y = np.asarray(y, float); w = np.ones(n) if w is None else w
        pbar = np.clip(np.average(y, weights=w), 1e-4, 1 - 1e-4)
        beta = np.zeros(p + 1); beta[0] = np.log(pbar / (1 - pbar))
        self.lam_trace = []
        for outer in range(self.max_outer if self.auto else 1):
            P = P0.copy()
            for (sl, S, r), l in zip(pens, lams): P[sl, sl] += l * S
            beta, A, ZtWZ, n_it = self._pirls(Z, y, w, P, beta)
            if not self.auto: break
            Ainv = np.linalg.inv(A); new = lams.copy()
            for i, ((sl, S, r), l) in enumerate(zip(pens, lams)):
                num = r - l * np.trace(Ainv[sl, sl] @ S); den = beta[sl] @ S @ beta[sl]
                upd = num / max(den, 1e-10) if num > 0 else l * 1e-1
                new[i] = np.clip(np.clip(upd, l / 10, l * 10), 1e-4, 1e7)   # damped generalised Fellner-Schall step
            self.lam_trace.append(lams.copy())
            live = np.minimum(new, lams) < 1e4   # terms already shrunk out (lambda > 1e4) are ignored in the test
            done = (not live.any()) or np.max(np.abs(np.log(new[live]) - np.log(lams[live]))) < 0.02
            lams = new
            if done: break
        self.lams = lams; self.n_outer = outer + 1
        self.beta = beta; self.n_iter = n_it
        H = np.linalg.solve(A, ZtWZ)
        self.edf = float(np.trace(H))   # effective degrees of freedom of the whole model
        k = 1; self.term_edf = {}
        for tm, s in zip(self.terms, sizes):
            self.term_edf[tm["col"]] = float(np.trace(H[k:k + s, k:k + s])); k += s
        return self

    def contributions(self, X):
        """Per-column additive contribution to the logit (n x n_cols) + intercept."""
        B, _ = self._basis(X)
        C = np.zeros((len(X), len(self.terms))); k = 0
        for j, s in enumerate(self.sizes):
            C[:, j] = B[:, k:k + s] @ self.beta[1 + k:1 + k + s]; k += s
        return C, self.beta[0]

    def logit(self, X):
        C, b0 = self.contributions(X); return b0 + C.sum(1)

    def proba(self, X):
        return 1 / (1 + np.exp(-self.logit(X)))

    def shape(self, col, grid):
        j = [tm["col"] for tm in self.terms].index(col)
        Xg = pd.DataFrame({c: np.zeros(len(grid)) for c in self.cols}); Xg[col] = grid
        C, _ = self.contributions(Xg); return C[:, j]


LAM_GRID = [0.3, 3.0, 30.0, 300.0, 3000.0]


def fit_gam_tuned(Xtr, ytr, Xva, yva, w_tr=None, **kw):
    best = None
    kw = {**kw, "auto": False}
    for lam in LAM_GRID:
        m = PSplineGAM(lam=lam, lam_null=1e-3, **kw).fit(Xtr, ytr, w_tr)
        ll = log_loss(yva, np.clip(m.proba(Xva), 1e-9, 1 - 1e-9), labels=[0, 1])
        if best is None or ll < best[0]: best = (ll, lam)
    return best[1]


class TwoStageGAM:
    """Proposed: process-aligned two-stage GAM. Stage S: P(supplemental review); stage A: P(approve | screened)."""
    name = "PS-GAM (proposed)"

    def __init__(self, lam_s=None, lam_a=None, auto=True, **kw):
        self.lam_s, self.lam_a, self.auto, self.kw = lam_s, lam_a, auto, kw

    def _gam(self, lam):
        if self.auto: return PSplineGAM(auto=True, **self.kw)
        return PSplineGAM(lam=lam, lam_null=1e-3, auto=False, **self.kw)

    def tune(self, Xtr, ytr, Xva, yva):
        if self.auto:  # smoothing parameters are estimated inside fit() (Fellner-Schall)
            self.lam_s = self.lam_a = "auto"; return self
        ys_tr, ys_va = (ytr == 1).astype(int), (yva == 1).astype(int)
        self.lam_s = fit_gam_tuned(Xtr, ys_tr, Xva, ys_va, **self.kw)
        mtr, mva = ytr != 1, yva != 1
        self.lam_a = fit_gam_tuned(Xtr[mtr], (ytr[mtr] == 0).astype(int), Xva[mva], (yva[mva] == 0).astype(int), **self.kw)
        return self

    def fit(self, X, y):
        self.gs = self._gam(self.lam_s).fit(X, (y == 1).astype(int))
        m = y != 1
        self.ga = self._gam(self.lam_a).fit(X[m], (y[m] == 0).astype(int))
        return self

    def proba(self, X):
        ps, pa = self.gs.proba(X), self.ga.proba(X)
        return np.column_stack([(1 - ps) * pa, ps, (1 - ps) * (1 - pa)])

    def attributions(self, X, cols):
        Cs, _ = self.gs.contributions(X); Ca, _ = self.ga.contributions(X)
        return np.column_stack([np.abs(Cs[:, g].sum(1)) + np.abs(Ca[:, g].sum(1)) for g in group_cols(cols)])


class OvRGAM:
    """Ablation: flat one-vs-rest GAMs (not process aligned), normalised."""
    def __init__(self, **kw): self.kw = {"auto": True, **kw}

    def tune(self, Xtr, ytr, Xva, yva):
        if self.kw.get("auto", True):
            self.lams = [1.0, 1.0, 1.0]; return self
        self.lams = [fit_gam_tuned(Xtr, (ytr == k).astype(int), Xva, (yva == k).astype(int), **self.kw) for k in range(3)]
        return self

    def fit(self, X, y):
        self.ms = [PSplineGAM(lam=l, lam_null=(1.0 if self.kw['auto'] else 1e-3), **self.kw).fit(X, (y == k).astype(int)) for k, l in enumerate(self.lams)]
        return self

    def proba(self, X):
        P = np.column_stack([m.proba(X) for m in self.ms]); return P / P.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- baselines
class LogReg:
    name = "Logistic regression"

    def tune(self, Xtr, ytr, Xva, yva):
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            self.C = C; self.fit(Xtr, ytr); ll = log_loss(yva, self.proba(Xva), labels=[0, 1, 2])
            if best is None or ll < best[0]: best = (ll, C)
        self.C = best[1]; return self

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.m = LogisticRegression(C=self.C, max_iter=3000).fit(self.sc.transform(X), y)
        return self

    def proba(self, X): return self.m.predict_proba(self.sc.transform(X))

    def attributions(self, X, cols):
        Z = self.sc.transform(X)  # standardised -> training mean = 0 reference
        return np.column_stack([np.abs(Z[:, g] @ self.m.coef_[:, g].T).sum(1) for g in group_cols(cols)])


class LGBM:
    name = "LightGBM + TreeSHAP"

    def __init__(self, seed=SEED): self.seed = seed

    def params(self, n):
        return dict(objective="multiclass", n_estimators=n, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, num_threads=1,
                    random_state=self.seed, verbose=-1)

    def tune(self, Xtr, ytr, Xva, yva):
        m = lgb.LGBMClassifier(**self.params(1500)).fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        self.n_est = max(int(m.best_iteration_ or 1500), 20); return self

    def fit(self, X, y):
        self.m = lgb.LGBMClassifier(**self.params(self.n_est)).fit(X, y); return self

    def proba(self, X): return self.m.predict_proba(X)

    def attributions(self, X, cols):
        S = self.m.predict_proba(X, pred_contrib=True)  # n x (K*(p+1))
        p = X.shape[1]; S = S.reshape(len(X), 3, p + 1)[:, :, :p]
        return np.column_stack([np.abs(S[:, :, g].sum(2)).sum(1) for g in group_cols(cols)])


class Rules:
    """Manual screening rules (decision list) with thresholds grid-searched on the training fold for macro-F1."""
    name = "Manual screening rules"

    def tune(self, Xtr, ytr, Xva=None, yva=None):
        best = None
        for pv, xu, pen, doc in itertools.product([20, 25, 30, 40], [0.9, 1.0, 1.2, 1.5], [0.40, 0.45, 0.50],
                                                 [0.55, 0.60, 0.65, 0.70]):
            self.th = (pv, xu, pen, doc)
            f = f1_score(ytr, self.predict(Xtr), average="macro")
            if best is None or f > best[0]: best = (f, self.th)
        self.th = best[1]; return self

    def fit(self, X, y): return self

    def predict(self, X):
        pv, xu, pen, doc = self.th
        rev = (X["pv_system_kw_dc"] > pv) | (X["xfmr_util"] > xu) | (X["feeder_der_penetration"] > pen)
        rej = (X["inverter_certification=uncertified"] == 1) | (X["document_completeness"] < doc)
        return np.where(rev, 1, np.where(rej, 2, 0))

    def proba(self, X):
        return np.eye(3)[self.predict(X)]


# ----------------------------------------------------------------------------- decision layer + metrics
W_GRID = [1, 1.5, 2, 3, 4, 6]


def tune_weights(P, y):
    best = None
    for ws, wr in itertools.product(W_GRID, W_GRID):
        f = f1_score(y, np.argmax(P * [1, ws, wr], 1), average="macro")
        if best is None or f > best[0]: best = (f, (1, ws, wr))
    return np.array(best[1], float)


def tune_tau(P, y, far_max=0.02):
    """Largest coverage auto-approval threshold whose false-auto-approval rate on validation <= far_max."""
    best = 0.999
    for tau in np.arange(0.50, 0.999, 0.005):
        auto = P[:, 0] >= tau
        if auto.sum() >= 20 and (y[auto] != 0).mean() <= far_max:
            best = tau; break
    return float(best)


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def metrics(P, y, wts, tau, prob=True):
    yhat = np.argmax(P * wts, 1)
    out = dict(accuracy=accuracy_score(y, yhat), macro_f1=f1_score(y, yhat, average="macro"),
               balanced_acc=balanced_accuracy_score(y, yhat))
    for k, s in enumerate(["f1_approve", "f1_review", "f1_reject"]):
        out[s] = f1_score(y == k, yhat == k)
    if prob:
        Pc = np.clip(P, 1e-9, 1); Pc = Pc / Pc.sum(1, keepdims=True)
        out["log_loss"] = log_loss(y, Pc, labels=[0, 1, 2])
        out["brier"] = float(np.mean(np.sum((Pc - np.eye(3)[y]) ** 2, 1)))
        out["ece"] = ece(Pc, y)
        auto = P[:, 0] >= tau
    else:
        auto = yhat == 0
    out["automation_rate"] = float(auto.mean())
    out["false_auto_approval"] = float((y[auto] != 0).mean()) if auto.any() else 0.0
    out["missed_review_auto"] = float(((y == 1) & auto).sum() / max((y == 1).sum(), 1))
    return out, yhat


def make(name):
    return {"PS-GAM (proposed)": TwoStageGAM, "Logistic regression": LogReg, "LightGBM + TreeSHAP": LGBM,
            "Manual screening rules": Rules}[name]()


MODELS = ["PS-GAM (proposed)", "Logistic regression", "LightGBM + TreeSHAP", "Manual screening rules"]


def run_protocol(Xtr_raw, ytr, Xte_raw, yte, model_names, seed, prep_kw=None, gam_kw=None, factory=None):
    """Fit preprocessing on train, tune on an inner 80/20 split, refit on full train, evaluate on test."""
    prep = Prep(**(prep_kw or {})).fit(Xtr_raw); Xtr = prep.transform(Xtr_raw); Xte = prep.transform(Xte_raw)
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
    out = {}
    for name in model_names:
        t = time.time()
        m = factory(name) if factory else make(name)
        if gam_kw is not None and hasattr(m, "kw"): m.kw = gam_kw
        m.tune(Xtr.iloc[itr], ytr[itr], Xtr.iloc[iva], ytr[iva])
        prob = not isinstance(m, Rules)
        if prob:
            m.fit(Xtr.iloc[itr], ytr[itr]); Pva = m.proba(Xtr.iloc[iva])
            wts = tune_weights(Pva, ytr[iva]); tau = tune_tau(Pva, ytr[iva])
        else:
            wts, tau = np.ones(3), 1.0
        m.fit(Xtr, ytr); fit_s = time.time() - t
        t = time.time(); P = m.proba(Xte); pred_ms = (time.time() - t) / len(yte) * 1e3
        met, yhat = metrics(P, yte, wts, tau, prob)
        met.update(fit_s=fit_s, pred_ms_per_app=pred_ms, tau=tau, w_review=wts[1], w_reject=wts[2])
        out[name] = dict(model=m, P=P, yhat=yhat, met=met, wts=wts, tau=tau)
    return out, prep, Xtr, Xte


# ----------------------------------------------------------------------------- explanation evaluation
def deletion_aopc(model, X, ref, cols, order, K=8):
    """Delete (set to the training reference) the top-k feature groups in the given order; track the drop in
    the originally predicted class probability. Returns curve of mean probability drop for k=0..K."""
    groups = group_cols(cols)
    P0 = model.proba(X); c = P0.argmax(1); p0 = P0[np.arange(len(X)), c]
    curve = [0.0]; Xk = X.to_numpy().copy()
    for k in range(K):
        for i in range(len(X)):
            Xk[i, groups[order[i, k]]] = ref[groups[order[i, k]]]
        Pk = model.proba(pd.DataFrame(Xk, columns=cols))
        curve.append(float(np.mean(p0 - Pk[np.arange(len(X)), c])))
    return np.array(curve)


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    feat_df = df.drop(columns=LEAK)
    y = df["y"].to_numpy()
    M = {"dataset": {}, "cv": {}, "temporal": {}, "explain": {}, "ablation": {}, "sensitivity": {}}

    # ---------------- dataset summary
    yr = df["submitted_at"].dt.year
    M["dataset"] = dict(rows=len(df), cols=df.shape[1] - 1, n_2025=int((yr == 2025).sum()), n_2026=int((yr == 2026).sum()),
                        class_share={c: float((df.decision == c).mean()) for c in CLASSES},
                        class_share_2025={c: float((df.decision[yr == 2025] == c).mean()) for c in CLASSES},
                        class_share_2026={c: float((df.decision[yr == 2026] == c).mean()) for c in CLASSES},
                        missing={c: float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()},
                        penetration_mean_2025=float(df.feeder_der_penetration[yr == 2025].mean()),
                        penetration_mean_2026=float(df.feeder_der_penetration[yr == 2026].mean()),
                        feeders=int(df.feeder_id.nunique()))
    print("dataset", M["dataset"], flush=True)

    # ---------------- Protocol A: 5-fold stratified CV
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fold_models, fold_prep = [], [], []
    for f, (tr, te) in enumerate(skf.split(feat_df, y)):
        out, prep, _, _ = run_protocol(feat_df.iloc[tr], y[tr], feat_df.iloc[te], y[te], MODELS, SEED + f)
        for n, o in out.items(): rows.append(dict(fold=f, model=n, **o["met"]))
        fold_models.append({n: out[n]["model"] for n in MODELS}); fold_prep.append(prep)
        print(f"fold {f} done {time.time()-T0:.0f}s", {n: round(o['met']['macro_f1'], 4) for n, o in out.items()}, flush=True)
    cv = pd.DataFrame(rows); cv.to_csv(os.path.join(RES, "cv_fold_metrics.csv"), index=False)
    keys = ["accuracy", "macro_f1", "balanced_acc", "f1_review", "f1_reject", "log_loss", "brier", "ece",
            "automation_rate", "false_auto_approval", "fit_s", "pred_ms_per_app"]
    summ = cv.groupby("model")[keys].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]; summ = summ.loc[MODELS]
    summ.to_csv(os.path.join(RES, "table_cv_summary.csv"))
    M["cv"]["summary"] = json.loads(summ.to_json(orient="index"))
    g = cv[cv.model == MODELS[0]].sort_values("fold")
    M["cv"]["paired_ttest_macro_f1"] = {}
    for n in MODELS[1:]:
        b = cv[cv.model == n].sort_values("fold")
        t, p = stats.ttest_rel(g.macro_f1.to_numpy(), b.macro_f1.to_numpy())
        tl, pl = (stats.ttest_rel(g.log_loss.to_numpy(), b.log_loss.to_numpy()) if n != "Manual screening rules" else (np.nan, np.nan))
        M["cv"]["paired_ttest_macro_f1"][n] = dict(diff=float((g.macro_f1.to_numpy() - b.macro_f1.to_numpy()).mean()),
                                                   t=float(t), p=float(p), p_logloss=float(pl))

    # ---------------- Protocol B: temporal drift (train 2025 -> test 2026)
    tr, te = np.where(yr == 2025)[0], np.where(yr == 2026)[0]
    out, prep, Xtr, Xte = run_protocol(feat_df.iloc[tr], y[tr], feat_df.iloc[te], y[te], MODELS, SEED)
    yte = y[te]
    trows = []
    for n in MODELS:
        met = out[n]["met"]
        bs = [f1_score(yte[i], out[n]["yhat"][i], average="macro") for i in
              (rng.integers(0, len(yte), len(yte)) for _ in range(1000))]
        met["macro_f1_ci_lo"], met["macro_f1_ci_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        trows.append(dict(model=n, **met))
    tdf = pd.DataFrame(trows); tdf.to_csv(os.path.join(RES, "table_temporal_metrics.csv"), index=False)
    M["temporal"]["metrics"] = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in trows}
    # drift drop: CV (in-distribution) vs temporal
    M["temporal"]["macro_f1_drop_vs_cv"] = {n: float(summ.loc[n, "macro_f1_mean"] - M["temporal"]["metrics"][n]["macro_f1"]) for n in MODELS}
    # McNemar (exact binomial) proposed vs others
    cg = out[MODELS[0]]["yhat"] == yte; M["temporal"]["mcnemar"] = {}
    for n in MODELS[1:]:
        cb = out[n]["yhat"] == yte; b01, b10 = int((cg & ~cb).sum()), int((~cg & cb).sum())
        M["temporal"]["mcnemar"][n] = dict(gam_only_correct=b01, other_only_correct=b10,
                                           p=float(stats.binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0)
    gam = out[MODELS[0]]["model"]
    M["temporal"]["gam_smoothing"] = dict(edf_review=gam.gs.edf, edf_approve=gam.ga.edf,
                                          outer_iter_review=gam.gs.n_outer, outer_iter_approve=gam.ga.n_outer,
                                          term_edf_review=gam.gs.term_edf, term_edf_approve=gam.ga.term_edf)
    pd.DataFrame({"term": list(gam.gs.term_edf), "edf_review": list(gam.gs.term_edf.values()),
                  "edf_approve": [gam.ga.term_edf[k] for k in gam.gs.term_edf]}).to_csv(
        os.path.join(RES, "gam_term_edf.csv"), index=False)

    # threshold recalibration under drift: re-select tau on January-2026 decisions, evaluate on Feb-Jun 2026
    mon = df["submitted_at"].dt.month.to_numpy()[te]; jan, rest = mon == 1, mon > 1
    M["temporal"]["recalibration"] = {}
    for n in MODELS[:3]:
        P = out[n]["P"]; t_old = out[n]["tau"]; t_new = tune_tau(P[jan], yte[jan])
        r_ = {}
        for lab, tt in (("tau_2025", t_old), ("tau_jan2026", t_new)):
            auto = P[rest, 0] >= tt
            r_[lab] = dict(tau=tt, coverage=float(auto.mean()), far=float((yte[rest][auto] != 0).mean()))
        M["temporal"]["recalibration"][n] = r_
    M["temporal"]["recalibration_n"] = dict(jan=int(jan.sum()), feb_jun=int(rest.sum()))
    M["temporal"]["lr_C"] = out["Logistic regression"]["model"].C
    M["temporal"]["lgbm_n_estimators"] = out["LightGBM + TreeSHAP"]["model"].n_est
    M["temporal"]["rules_thresholds"] = list(map(float, out["Manual screening rules"]["model"].th))
    cm = confusion_matrix(yte, out[MODELS[0]]["yhat"], labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(RES, "confusion_gam_temporal.csv"))
    M["temporal"]["confusion_gam"] = cm.tolist()
    cmb = confusion_matrix(yte, out["LightGBM + TreeSHAP"]["yhat"], labels=[0, 1, 2])
    M["temporal"]["confusion_lgbm"] = cmb.tolist()
    print("temporal done", f"{time.time()-T0:.0f}s", tdf[["model", "macro_f1", "log_loss", "automation_rate", "false_auto_approval"]].to_string(), flush=True)

    # selective-automation curves (coverage vs false-auto-approval) on temporal test
    cov_rows = []
    for n in MODELS[:3]:
        P = out[n]["P"]
        for tau in np.round(np.arange(0.50, 0.996, 0.005), 3):
            auto = P[:, 0] >= tau
            cov_rows.append(dict(model=n, tau=tau, coverage=auto.mean(),
                                 far=(yte[auto] != 0).mean() if auto.any() else 0.0))
    covdf = pd.DataFrame(cov_rows); covdf.to_csv(os.path.join(RES, "automation_curves_temporal.csv"), index=False)
    # coverage at a fixed 2% FAR (oracle on test, reported only for the curve comparison)
    M["temporal"]["coverage_at_far2_curve"] = {n: float(covdf[(covdf.model == n) & (covdf.far <= 0.02)].coverage.max()) for n in MODELS[:3]}
    rm = out["Manual screening rules"]["met"]
    M["temporal"]["rules_point"] = dict(coverage=rm["automation_rate"], far=rm["false_auto_approval"])

    # ---------------- explanation evaluation on temporal test (1,000 sampled applications)
    cols = list(Xtr.columns); sub = rng.choice(len(yte), 1000, replace=False); Xs = Xte.iloc[sub].reset_index(drop=True)
    ref = Xtr.median().to_numpy()
    for c in cols:
        if "=" in c or c in (INTER, "battery_attached"): ref[cols.index(c)] = Xtr[c].mode().iloc[0]
    ex_rows, curves = [], {}
    roof = GROUPS.index("roof_age_years")
    for n in MODELS[:3]:
        m = out[n]["model"]; t = time.time(); A = m.attributions(Xs, cols); ex_ms = (time.time() - t) / len(Xs) * 1e3
        order = np.argsort(-A, 1); rnd = np.array([rng.permutation(len(GROUPS)) for _ in range(len(Xs))])
        cur = deletion_aopc(m, Xs, ref, cols, order); cur_r = deletion_aopc(m, Xs, ref, cols, rnd)
        curves[n] = cur; curves[n + " (random)"] = cur_r
        share = A / A.sum(1, keepdims=True)
        top3 = np.argsort(-A, 1)[:, :3]
        ex_rows.append(dict(model=n, aopc=float(cur[1:].mean()), aopc_random=float(cur_r[1:].mean()),
                            distractor_share=float(share[:, roof].mean()),
                            distractor_in_top3=float((top3 == roof).any(1).mean()),
                            n_terms_90=float(np.mean((np.cumsum(-np.sort(-share, 1), 1) < 0.9).sum(1) + 1)),
                            explain_ms_per_app=ex_ms))
    # cross-fold stability of global importance and per-application top-3 (fold models, same evaluation rows)
    for r in ex_rows:
        n = r["model"]; G, T3 = [], []
        for fm, fp in zip(fold_models, fold_prep):
            Xf = fp.transform(feat_df.iloc[te].iloc[sub]).reset_index(drop=True)
            A = fm[n].attributions(Xf, cols); G.append(A.mean(0)); T3.append(np.argsort(-A, 1)[:, :3])
        rho = [stats.spearmanr(G[i], G[j]).correlation for i in range(5) for j in range(i + 1, 5)]
        jac = [np.mean([len(set(T3[i][k]) & set(T3[j][k])) / len(set(T3[i][k]) | set(T3[j][k])) for k in range(len(sub))])
               for i in range(5) for j in range(i + 1, 5)]
        r["global_rank_stability"] = float(np.mean(rho)); r["top3_jaccard_stability"] = float(np.mean(jac))
    exdf = pd.DataFrame(ex_rows); exdf.to_csv(os.path.join(RES, "table_explanation_metrics.csv"), index=False)
    M["explain"]["metrics"] = {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in ex_rows}
    pd.DataFrame({k: v for k, v in curves.items()}).rename_axis("k_deleted").to_csv(os.path.join(RES, "deletion_curves.csv"))
    # global importance of the proposed model (mean |contribution| per stage)
    Cs, _ = gam.gs.contributions(Xte); Ca, _ = gam.ga.contributions(Xte)
    gi = pd.DataFrame({"group": GROUPS, "review_stage": [np.abs(Cs[:, g].sum(1)).mean() for g in group_cols(cols)],
                       "approve_stage": [np.abs(Ca[:, g].sum(1)).mean() for g in group_cols(cols)]})
    gi.to_csv(os.path.join(RES, "gam_global_importance.csv"), index=False)
    M["explain"]["gam_global_importance"] = gi.set_index("group").to_dict(orient="index")
    print("explain done", f"{time.time()-T0:.0f}s", exdf.to_string(), flush=True)

    # example reason codes for one borderline application routed to a human
    P = out[MODELS[0]]["P"]; border = np.where((P[:, 0] < out[MODELS[0]]["tau"]) & (P[:, 0] > 0.55) & (yte == 2))[0]
    i0 = int(border[0]) if len(border) else int(np.argmin(np.abs(P[:, 0] - 0.7)))
    xi = Xte.iloc[[i0]]; ca, b0a = gam.ga.contributions(xi); cs, b0s = gam.gs.contributions(xi)
    gc = group_cols(cols)
    rc = pd.DataFrame({"group": GROUPS, "value": [", ".join(f"{xi.iloc[0, j]:.3g}" for j in g) for g in gc],
                       "approve_logit": [ca[0, g].sum() for g in gc], "review_logit": [cs[0, g].sum() for g in gc]})
    rc = rc.reindex(rc.approve_logit.abs().sort_values(ascending=False).index)
    rc.to_csv(os.path.join(RES, "example_reason_codes.csv"), index=False)
    raw = df.iloc[te].iloc[i0]
    M["explain"]["example"] = dict(application_id=raw.application_id, true=raw.decision,
                                   p=list(map(float, P[i0])), intercept_approve=float(b0a), intercept_review=float(b0s),
                                   top=rc.head(6).to_dict(orient="records"))

    # counterfactual recourse: predicted reject/rework -> minimal document completeness (and certified inverter)
    pred = out[MODELS[0]]["yhat"]; wts = out[MODELS[0]]["wts"]; rej = np.where(pred == 2)[0]
    cf_doc, cf_inv = [], []
    for i in rej:
        xi = Xte.iloc[[i]].copy(); base = xi["document_completeness"].iloc[0]; found = None
        for d in np.arange(base, 1.0001, 0.01):
            xi["document_completeness"] = d
            if np.argmax(gam.proba(xi)[0] * wts) == 0: found = d - base; break
        cf_doc.append(found)
        xj = Xte.iloc[[i]].copy()
        if xj["inverter_certification=uncertified"].iloc[0] == 1 or xj["inverter_certification=ul1741sa_legacy"].iloc[0] == 1:
            xj["inverter_certification=uncertified"] = 0; xj["inverter_certification=ul1741sa_legacy"] = 0
            cf_inv.append(int(np.argmax(gam.proba(xj)[0] * wts) == 0))
    fd = [v for v in cf_doc if v is not None]
    M["explain"]["counterfactual"] = dict(n_pred_reject=int(len(rej)), doc_feasible_share=len(fd) / max(len(rej), 1),
                                          doc_median_increase=float(np.median(fd)) if fd else None,
                                          doc_iqr=[float(np.percentile(fd, 25)), float(np.percentile(fd, 75))] if fd else None,
                                          n_noncert=len(cf_inv), inverter_upgrade_flip_share=float(np.mean(cf_inv)) if cf_inv else None)
    print("counterfactual", M["explain"]["counterfactual"], flush=True)

    # ---------------- ablation (5-fold CV, proposed variants)
    def fac(variant):
        def f(name):
            if variant == "full": return TwoStageGAM()
            if variant == "global": return TwoStageGAM(auto=False)
            if variant == "flat": return OvRGAM()
            if variant == "linear": return TwoStageGAM(linear_only=tuple(SPLINE))
            if variant == "noscreen": return TwoStageGAM(linear_only=())
            if variant == "nowinsor": return TwoStageGAM()
        return f
    variants = [("full", "PS-GAM (full)", {}), ("global", "Single global lambda (grid)", {}),
                ("flat", "Flat one-vs-rest GAM", {}),
                ("linear", "Two-stage, linear terms only", {}), ("noscreen", "Two-stage GAM, no screen features", {}),
                ("nowinsor", "Two-stage GAM, no winsorisation", {"winsor": False})]
    ab_rows = []
    for f, (trf, tef) in enumerate(skf.split(feat_df, y)):
        for key, label, pk in variants:
            Xa, Xb = feat_df.iloc[trf], feat_df.iloc[tef]
            if key == "noscreen":
                # remove the three screen features by zeroing them (constant -> no effect, term dropped)
                class P2(Prep):
                    def transform(self, d):
                        X = super().transform(d)
                        for c in SCREEN: X[c] = 0.0
                        return X
                prep = P2().fit(Xa); Xa_t, Xb_t = prep.transform(Xa), prep.transform(Xb)
                Xa_t = Xa_t.drop(columns=SCREEN); Xb_t = Xb_t.drop(columns=SCREEN)
                ytr_, yte_ = y[trf], y[tef]
                itr, iva = train_test_split(np.arange(len(ytr_)), test_size=0.2, stratify=ytr_, random_state=SEED + f)
                m = TwoStageGAM().tune(Xa_t.iloc[itr], ytr_[itr], Xa_t.iloc[iva], ytr_[iva]).fit(Xa_t.iloc[itr], ytr_[itr])
                Pva = m.proba(Xa_t.iloc[iva]); wts = tune_weights(Pva, ytr_[iva]); tau = tune_tau(Pva, ytr_[iva])
                m.fit(Xa_t, ytr_); met, _ = metrics(m.proba(Xb_t), yte_, wts, tau)
            else:
                o, _, _, _ = run_protocol(Xa, y[trf], Xb, y[tef], ["x"], SEED + f, prep_kw=pk, factory=fac(key))
                met = o["x"]["met"]
            ab_rows.append(dict(fold=f, variant=label, **met))
        print(f"ablation fold {f} {time.time()-T0:.0f}s", flush=True)
    ab = pd.DataFrame(ab_rows)
    abs_ = ab.groupby("variant")[["macro_f1", "f1_review", "f1_reject", "log_loss", "automation_rate", "false_auto_approval"]].agg(["mean", "std"])
    abs_.columns = [f"{a}_{b}" for a, b in abs_.columns]; abs_ = abs_.loc[[v[1] for v in variants]]
    ab.to_csv(os.path.join(RES, "ablation_folds.csv"), index=False); abs_.to_csv(os.path.join(RES, "table_ablation.csv"))
    M["ablation"] = json.loads(abs_.to_json(orient="index"))
    full = ab[ab.variant == "PS-GAM (full)"].sort_values("fold")
    M["ablation_p"] = {}
    for _, label, _ in variants[1:]:
        b = ab[ab.variant == label].sort_values("fold")
        M["ablation_p"][label] = dict(p_macro_f1=float(stats.ttest_rel(full.macro_f1, b.macro_f1).pvalue),
                                      p_log_loss=float(stats.ttest_rel(full.log_loss, b.log_loss).pvalue))

    # ---------------- smoothing sensitivity (temporal split, fixed lambda for both stages)
    sens = []
    itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y[tr], random_state=SEED)
    for lam in [0.03, 0.3, 3.0, 30.0, 300.0, 3000.0, 30000.0]:
        m = TwoStageGAM(lam_s=lam, lam_a=lam, auto=False).fit(Xtr.iloc[itr], y[tr][itr])
        Pva = m.proba(Xtr.iloc[iva]); wts = tune_weights(Pva, y[tr][iva]); tau = tune_tau(Pva, y[tr][iva])
        m.fit(Xtr, y[tr]); met, _ = metrics(m.proba(Xte), yte, wts, tau)
        sens.append(dict(lam=lam, edf_review=m.gs.edf, edf_approve=m.ga.edf, **met))
    sdf = pd.DataFrame(sens); sdf.to_csv(os.path.join(RES, "smoothing_sensitivity.csv"), index=False)
    M["sensitivity"] = sdf.to_dict(orient="records")
    # threshold sensitivity for the proposed model on temporal test
    thr = covdf[covdf.model == MODELS[0]]
    M["sensitivity_tau"] = thr[thr.tau.isin([0.8, 0.85, 0.9, 0.95, 0.97, 0.99])].to_dict(orient="records")

    # ---------------- figures
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    colors = {"PS-GAM (proposed)": "#c0392b", "Logistic regression": "#7f8c8d", "LightGBM + TreeSHAP": "#34495e",
              "Manual screening rules": "#bdc3c7"}

    def save(fig, name):
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 92})
        plt.close(fig)

    # Fig 2: shape functions of the approve stage (and review stage) for key screens + distractor
    feats = ["remaining_hc_kw", "xfmr_util", "document_completeness", "service_ratio", "installer_quality_score", "roof_age_years"]
    units = {"remaining_hc_kw": "Remaining hosting capacity (kW)", "xfmr_util": "Transformer utilisation (p.u.)",
             "document_completeness": "Document completeness (0-1)", "service_ratio": "PV / service rating (kW/kVA)",
             "installer_quality_score": "Installer score (0-1)", "roof_age_years": "Roof age (years)"}
    fig, axs = plt.subplots(3, 2, figsize=(3.4, 3.9))
    for ax, c in zip(axs.ravel(), feats):
        grid = np.linspace(Xtr[c].quantile(0.01), Xtr[c].quantile(0.99), 200)
        ax.plot(grid, gam.ga.shape(c, grid), color="#c0392b", lw=1.4, label="Approve stage $f^A_j$")
        ax.plot(grid, gam.gs.shape(c, grid), color="#34495e", lw=1.2, ls="--", label="Review stage $f^S_j$")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlabel(units[c], fontsize=7); ax.set_ylabel("Logit contrib.", fontsize=7)
        ax.tick_params(labelsize=6.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.5, rect=(0, 0, 1, 0.95))
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig2_shape_functions")

    # Fig 3: CV + temporal macro-F1 and log-loss comparison
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0))
    lab = ["PS-GAM", "LR", "LGBM", "Rules"]; x = np.arange(4)
    for ax, k, ttl in [(axs[0], "macro_f1", "Macro-F1"), (axs[1], "automation_rate", "Automation rate")]:
        mu = [summ.loc[n, f"{k}_mean"] for n in MODELS]; sd = [summ.loc[n, f"{k}_std"] for n in MODELS]
        tv = [M["temporal"]["metrics"][n][k] for n in MODELS]
        ax.bar(x - 0.2, mu, 0.38, yerr=sd, capsize=2, color=[colors[n] for n in MODELS], label="5-fold CV")
        ax.bar(x + 0.2, tv, 0.38, color=[colors[n] for n in MODELS], alpha=0.45, hatch="///", edgecolor="k", lw=0.3,
               label="2026 drift test")
        ax.set_xticks(x); ax.set_xticklabels(lab, rotation=35, ha="right", rotation_mode="anchor", fontsize=6.5)
        ax.set_ylabel(ttl, labelpad=1)
        lo = min(min(mu), min(tv)); ax.set_ylim(max(0, lo - 0.1), min(1.0, max(max(mu), max(tv)) + 0.08))
    axs[0].legend(fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout(pad=0.3, w_pad=1.2); save(fig, "fig3_comparison")

    # Fig 4: selective automation curves
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for n in MODELS[:3]:
        d = covdf[covdf.model == n]; ax.plot(d.coverage * 100, d.far * 100, color=colors[n], lw=1.4, label=n)
        mt = M["temporal"]["metrics"][n]
        ax.plot(mt["automation_rate"] * 100, mt["false_auto_approval"] * 100, "o", color=colors[n], ms=4)
    ax.plot(rm["automation_rate"] * 100, rm["false_auto_approval"] * 100, "s", color="#7f7f7f", ms=5, label="Manual screening rules")
    ax.axhline(2, color="0.6", ls=":", lw=0.8); ax.text(2, 2.15, "target 2 %", fontsize=6.5, color="0.4")
    ax.set_xlabel("Applications auto-approved (%)"); ax.set_ylabel("False auto-approvals (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3); save(fig, "fig4_automation_tradeoff")

    # Fig 5: deletion faithfulness curves
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for n in MODELS[:3]:
        ax.plot(range(9), curves[n], color=colors[n], lw=1.5, marker="o", ms=3, label=n)
        ax.plot(range(9), curves[n + " (random)"], color=colors[n], lw=1.0, ls="--")
    ax.set_xlabel("Feature groups deleted, k"); ax.set_ylabel("Drop in predicted-class prob.")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", title="solid: attribution order; dashed: random",
              title_fontsize=6.5)
    ax.set_ylim(0, max(max(c) for c in curves.values()) * 1.45)
    fig.tight_layout(pad=0.3); save(fig, "fig5_deletion_faithfulness")

    # Fig 6: confusion matrices (proposed vs LightGBM) on 2026 test
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.75))
    for ax, cmx, ttl in [(axs[0], cm, "PS-GAM (proposed)"), (axs[1], cmb, "LightGBM")]:
        cn = cmx / cmx.sum(1, keepdims=True)
        ax.imshow(cn, cmap="Reds" if "GAM" in ttl else "Greys", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmx[i, j]}\n{cn[i, j]*100:.0f}%", ha="center", va="center", fontsize=6,
                        color="white" if cn[i, j] > 0.6 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["A", "S", "R"]); ax.set_yticklabels(["A", "S", "R"])
        ax.set_xlabel("Predicted"); ax.set_title(ttl, fontsize=7)
    axs[0].set_ylabel("Engineer decision")
    fig.tight_layout(pad=0.3); save(fig, "fig6_confusion")

    # Fig 1 (methodology flowchart, Graphviz)
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join(HERE, "make_fig1.py")], check=True)

    M["runtime_s"] = time.time() - T0
    with open(os.path.join(RES, "metrics.json"), "w") as fh:
        json.dump(M, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"ALL DONE in {M['runtime_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
