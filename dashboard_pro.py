# ==========================================================
# ALADDIN 3.2 — MAXI HEDGEFUND (Single-file Streamlit App)
# Minimal Neon UI • Multi-Optimizer • Black-Litterman (cost-aware)
# Walk-Forward (TC/Liquidity/Almgren–Chriss) • EWMA/Regime 2.0
# Target-Vol/Lev + Drawdown-Guard + Hedge Overlay (GLD/USD)
# Momentum Tilt • Purged K-Fold CV • Deflated Sharpe • PBO
# FX Base (EUR/USD/CHF/GBP) + FX-Hedge • Risk Budgeting
# Stress (2008/2020/2022) • Rebalance Planner • Paper Broker
# Presets • Report • AI-Insights (light) • Ensemble Blending
#
# NEW (3.2):
# - Optimizer-Ensemble (MinVol, HRP, Min-CVaR, BL, ERC) + CV-basiertes Blending
# - ERC (Equal Risk Contribution) & Risk-Budgeting (Gruppen-Risikobeiträge)
# - Regime-Engine 2.0 (vol/mom/corr) → Modellwahl + Vol-Target-Scaler
# - Drawdown-Guard (adaptive Leverage) + optional Hedge-Overlay (GLD/FX)
# - FX-Basis & FX-Hedge (EUR/USD/CHF/GBP) via FX-Serien (EURUSD=X etc.)
# - PBO/Reality-Check für Strategie-Selektion (Ensemble vs. Einzel)
# - Almgren–Chriss light (temp & perm Impact) in Walk-Forward-TC
# - Noch robustere Loader + I/O + saubere Fallbacks
# ==========================================================

import io, json, math, time, warnings, random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------- Page / Theme ----------
st.set_page_config(page_title="ALADDIN 3.2 — MAXI HEDGEFUND", layout="wide")

pio.templates["aladdin"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, Segoe UI, Roboto, sans-serif", size=14),
        paper_bgcolor="#0c0f14",
        plot_bgcolor="#0c0f14",
        colorway=["#22d3ee", "#14b8a6", "#f59e0b", "#f43f5e", "#a78bfa", "#60a5fa"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
)
pio.templates.default = "aladdin"
PLOTLY_CFG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format":"png",
        "filename":"aladdin_chart",
        "height":450,
        "width":900,
        "scale":2
    }
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }

/* Minimal Neon Background: mesh + grid + glow */
body {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,.08), transparent 60%),
    radial-gradient(1200px 600px at 100% 0%, rgba(167,139,250,.07), transparent 60%),
    radial-gradient(1200px 600px at 50% 120%, rgba(20,184,166,.08), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)),
    #0c0f14;
}
:root { --grid-color: rgba(255,255,255,0.04); --card: rgba(18,22,30,.8); --border: rgba(255,255,255,.08); }
.background-grid:before {
  content:""; position: fixed; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(var(--grid-color) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
  background-size: 34px 34px, 34px 34px; opacity:.7;
  mask-image: radial-gradient(circle at 50% 40%, rgba(255,255,255,.35), transparent 55%);
}

/* Layout polish */
.block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
h1,h2,h3 { letter-spacing:.3px; }

/* Glass KPI cards with neon glow on hover */
.kpi-card {
  background: var(--card); border:1px solid var(--border);
  border-radius:14px; padding:16px 18px; box-shadow:0 12px 30px rgba(0,0,0,.30);
  transition: box-shadow .25s ease, border-color .25s ease;
}
.kpi-card:hover { border-color: rgba(34,211,238,.45); box-shadow: 0 0 0 1px rgba(34,211,238,.25), 0 20px 40px rgba(0,0,0,.35); }
.kpi-big { font-size:2.0rem; font-weight:800; color:#eaf7ff; line-height:1.1; }
.kpi-label { color:#95a3b3; font-size:.92rem; }
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent); margin:12px 0 18px; }
.smallnote { color:#93a1b1; font-size:.9rem; }

/* DataFrames & Buttons */
.stDataFrame { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.stButton>button { border-radius:10px; }

/* Tabs: sticky + pill look */
.stTabs [data-baseweb="tab-list"] { gap: 6px; position: sticky; top: 0; z-index: 10; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.06); border-radius: 10px; padding: 8px 12px; }
.stTabs [aria-selected="true"] { background: rgba(34,211,238,.14); border:1px solid rgba(34,211,238,.35); }

/* Card section */
.section { border:1px solid var(--border); border-radius:14px; padding:16px; background: var(--card); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="background-grid"></div>', unsafe_allow_html=True)

# ---------- Assets & FX ----------
ASSET_MAP: Dict[str, str] = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Gold": "GLD",
    "NVIDIA": "NVDA",
    "NASDAQ": "QQQ",        # Nasdaq-100 proxy
    "S&P 500": "SPY",       # S&P 500 proxy
    "MSCI World ETF": "URTH",
    "STARLINK (SPACE X)": "UFO",  # Procure Space ETF
}
FRIENDLY_ORDER = list(ASSET_MAP.keys())

# Group tags (extend as you like)
CRYPTOS = {"BTC-USD","ETH-USD","SOL-USD"}
EQUITIES = {"AAPL","TSLA","NVDA","QQQ","SPY","URTH","UFO"}
HEDGE_TICKERS = {"GLD"}  # gold overlay candidate

# FX symbols (yfinance):
FX_BASES = {
    "EUR": "EURUSD=X",  # USD per EUR
    "USD": None,
    "CHF": "CHFUSD=X",
    "GBP": "GBPUSD=X"
}

# ---------- Utils & Risk ----------
def set_global_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)
set_global_seed(42)

def to_returns(px: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    px = px.sort_index()
    rets = np.log(px/px.shift(1)) if log else px.pct_change()
    return rets.dropna(how="all").fillna(0.0)

def annualize_stats(series: pd.Series, freq: int = 252) -> Tuple[float, float]:
    if series.empty: return 0.0, 0.0
    mu  = float(series.mean()) * freq
    vol = float(series.std(ddof=0)) * math.sqrt(freq)
    return mu, vol

def sharpe_ratio(series: pd.Series, rf: float = 0.0, freq: int = 252) -> float:
    if series.empty: return 0.0
    ex  = series - (rf / freq)
    vol = float(ex.std(ddof=0))
    return 0.0 if vol == 0 else float(ex.mean() / vol * math.sqrt(freq))

def hist_var_es(series: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    s = np.sort(series.dropna().values)
    if s.size == 0: return 0.0, 0.0
    k = max(0, int((1.0 - alpha) * len(s)) - 1)
    var = s[k]
    es  = s[:k+1].mean() if k >= 0 else var
    return float(var), float(es)

def higher_moments(series: pd.Series) -> Tuple[float,float]:
    if series.empty: return 0.0, 0.0
    x = series.dropna()
    if x.std(ddof=0) == 0: return 0.0, 0.0
    skew = float(((x - x.mean())**3).mean() / (x.std(ddof=0)**3))
    kurt = float(((x - x.mean())**4).mean() / (x.std(ddof=0)**4)) - 3.0
    return skew, kurt

def max_drawdown(cum: pd.Series) -> float:
    if cum.empty: return 0.0
    dd = (cum / cum.cummax()) - 1.0
    return float(dd.min())

def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    X = returns.fillna(0.0).values
    n, k = X.shape
    seed = min(252, max(60, n//4))
    S = np.cov(X[:seed].T) if n >= seed else np.eye(k)
    for t in range(seed, n):
        x = X[t].reshape(-1,1)
        S = lam * S + (1 - lam) * (x @ x.T)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def corr_regime(returns: pd.DataFrame, lookback:int=63) -> float:
    """
    Mittlere paarweise Korrelation (Upper Triangle) – robust bei kurzer History.
    Gibt 0.0 zurück, wenn zu wenige Daten vorhanden sind.
    """
    if returns is None or returns.empty:
        return 0.0
    lb = min(lookback, len(returns))
    if lb < 3:
        return 0.0
    c = returns.tail(lb).corr().values
    if c.size == 0:
        return 0.0
    iu = np.triu_indices_from(c, 1)
    return float(np.nanmean(c[iu])) if len(iu[0]) else 0.0


def regime_tag(returns: pd.DataFrame,
               lb_vol:int=21, lb_mom:int=63,
               thr_vol:float=0.025, thr_corr:float=0.40) -> str:
    """
    Regime-Engine (robust):
    - Volatilität: rolling Std. über Zeit, am Ende querschnittlich gemittelt
    - Momentum: rolling Mittel der *querschnittlichen Tagesrendite*
    - Korrelation: mittlere paarweise Korrelation
    Mit sauberen Fallbacks, damit kein .iloc[-1]-Crash auftreten kann.
    """
    if returns is None or returns.empty or len(returns) < 3:
        return "unknown"

    # Volatilität (zeitlich je Asset) → Querschnittsmittel am letzten verfügbaren Tag
    vol_df = returns.rolling(lb_vol).std()
    if vol_df.dropna(how="all").shape[0] > 0 and not vol_df.dropna(how="all").tail(1).empty:
        vol = float(vol_df.dropna(how="all").tail(1).mean(axis=1).iloc[0])
    else:
        vol = float(returns.std(ddof=0).mean())  # Fallback

    # Cross-sectional daily mean → Momentum über Zeit
    cs_mean = returns.mean(axis=1)  # Serie über Datum
    mom_roll = cs_mean.rolling(lb_mom).mean()
    if mom_roll.dropna().shape[0] > 0 and not mom_roll.dropna().tail(1).empty:
        mom = float(mom_roll.dropna().tail(1).iloc[0])
    else:
        mom = float(cs_mean.mean())  # Fallback

    # Korrelation
    acorr = corr_regime(returns, lookback=lb_mom)

    # Heuristik
    if vol > thr_vol and acorr > thr_corr:
        return "high-vol"
    if mom < 0:
        return "bear"
    return "normal"

def sparkline(series: pd.Series, height=60):
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines"))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=height, template="aladdin")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

# ---------- Robust Data Loader ----------
def _download_with_retry(tickers, start=None, period=None, tries=3, sleep=1.2):
    last_err = None
    for k in range(tries):
        try:
            df = yf.download(
                tickers,
                start=start, period=period,
                auto_adjust=True, progress=False,
                group_by="ticker", threads=True
            )
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                return df
        except Exception as e:
            last_err = e
        time.sleep(sleep * (k+1))
    if last_err: raise last_err
    return pd.DataFrame()

def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        px = df["Adj Close"].copy()
    elif isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        px = df.xs("Adj Close", axis=1, level=1) if "Adj Close" in lvl1 else df.xs("Close", axis=1, level=1)
    else:
        px = df if isinstance(df, pd.DataFrame) else df.to_frame()
    if isinstance(px, pd.Series): px = px.to_frame()
    return px

@st.cache_data(show_spinner=False, ttl=300)
def load_prices(tickers: List[str], years: int = 5, clamp_thr: float=0.40) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    raw = _download_with_retry(tickers, start=start)
    px = _normalize_price_frame(raw)
    px = px.reindex(columns=[t for t in tickers if t in px.columns]).ffill().bfill()
    pct = px.pct_change()
    bad = pct.abs() > clamp_thr
    if bad.any().any():
        px = px.copy()
        for col in px.columns:
            for t in bad.index[bad[col]]:
                prev = px[col].shift(1).get(t, np.nan)
                nxt  = px[col].shift(-1).get(t, np.nan)
                rep = np.nanmean([prev, nxt])
                if not np.isnan(rep): px.at[t, col] = rep
    return px.dropna(how="all")

@st.cache_data(show_spinner=False, ttl=300)
def load_ohlcv(tickers: List[str], years: int = 5, clamp_thr: float=0.40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = (datetime.today() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    df = _download_with_retry(tickers, start=start)
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        px = df.xs("Adj Close", axis=1, level=1) if "Adj Close" in df.columns.get_level_values(1) else df.xs("Close", axis=1, level=1)
        vol = df.xs("Volume", axis=1, level=1)
    else:
        px = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        vol = df["Volume"] if "Volume" in df.columns else pd.DataFrame(index=px.index)
    px = px.reindex(columns=[t for t in tickers if t in px.columns]).ffill().bfill()
    vol = vol.reindex(columns=[t for t in tickers if t in vol.columns]).fillna(0)
    pct = px.pct_change(); bad = pct.abs() > clamp_thr
    if bad.any().any():
        px = px.copy()
        for col in px.columns:
            for t in bad.index[bad[col]]:
                prev = px[col].shift(1).get(t, np.nan)
                nxt  = px[col].shift(-1).get(t, np.nan)
                rep = np.nanmean([prev, nxt])
                if not np.isnan(rep): px.at[t, col] = rep
    return px.dropna(how="all"), vol.fillna(0)

# ---------- Optimizers ----------
def _project_to_simplex_with_bounds(w: np.ndarray, lb: float, ub: float) -> np.ndarray:
    w = np.clip(w, lb, ub); s = w.sum()
    if s <= 0: w[:] = 1.0 / len(w)
    else:
        w = w / s
        w = np.clip(w, lb, ub)
        w = w / w.sum()
    return w

def inverse_variance_weights(px_hist: pd.DataFrame, lb: float, ub: float) -> pd.Series:
    n = px_hist.shape[1]
    if n == 0: return pd.Series(dtype=float)
    if px_hist.shape[0] < 40: return pd.Series(np.ones(n)/n, index=px_hist.columns, dtype=float)
    rets = to_returns(px_hist)
    var = rets.var().replace([np.inf, -np.inf], np.nan).fillna(1.0)
    iv = 1.0 / (var + 1e-12)
    w = _project_to_simplex_with_bounds(iv.values, lb, ub)
    return pd.Series(w, index=px_hist.columns, dtype=float)

def optimizer_min_vol(px_hist: pd.DataFrame, lb: float, ub: float, use_ewma: bool) -> Tuple[pd.Series, str]:
    try:
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.efficient_frontier import EfficientFrontier
        if use_ewma: S = ewma_cov(to_returns(px_hist), lam=0.94)
        else:
            from pypfopt.risk_models import CovarianceShrinkage
            S = CovarianceShrinkage(px_hist).ledoit_wolf()
        mu = mean_historical_return(px_hist)
        ef = EfficientFrontier(mu, S, weight_bounds=(lb, ub))
        w = pd.Series(ef.min_volatility()); w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0);  w = w / w.sum()
        return w, ("Min-Vol (EWMA)" if use_ewma else "Min-Vol (Ledoit-Wolf)")
    except Exception:
        return inverse_variance_weights(px_hist, lb, ub), "Inverse-Variance"

def optimizer_hrp(px_hist: pd.DataFrame, lb: float, ub: float) -> Tuple[pd.Series, str]:
    try:
        from pypfopt.hierarchical_risk_parity import HRPOpt
        rets = to_returns(px_hist)
        hrp = HRPOpt(rets)
        w = pd.Series(hrp.optimize()); w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0); w = pd.Series(_project_to_simplex_with_bounds(w.values, lb, ub), index=w.index)
        return w, "HRP"
    except Exception:
        return inverse_variance_weights(px_hist, lb, ub), "Inverse-Variance"

def optimizer_cvar(returns: pd.DataFrame, alpha: float, lb: float, ub: float, allow_short: bool) -> Tuple[pd.Series, str]:
    try:
        import cvxpy as cp
        R = returns.values; T, N = R.shape
        w = cp.Variable(N); z = cp.Variable(T); v = cp.Variable(1)
        tau = 1 - alpha
        obj = v + (1.0/(tau*T)) * cp.sum(z)
        cons = []
        if allow_short: cons += [cp.sum(cp.abs(w)) == 1.0, w >= -ub, w <= ub]
        else: cons += [cp.sum(w) == 1.0, w >= lb, w <= ub]
        cons += [z >= 0, z >= -(R @ w) - v]
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
        w_opt = pd.Series(np.array(w.value).ravel(), index=returns.columns).fillna(0.0)
        w_opt = pd.Series(_project_to_simplex_with_bounds(w_opt.values, lb, ub), index=w_opt.index)
        return w_opt, "Min-CVaR"
    except Exception:
        return inverse_variance_weights(returns.cumsum(), lb, ub), "Inverse-Variance"

# --- ERC (Equal Risk Contribution) ---
def optimizer_erc(px_hist: pd.DataFrame, lb: float, ub: float, use_ewma: bool=True, iters:int=500) -> Tuple[pd.Series, str]:
    rets = to_returns(px_hist)
    S = ewma_cov(rets) if use_ewma else rets.cov()
    n = S.shape[0]
    if n == 0: return pd.Series(dtype=float), "ERC"
    w = np.ones(n)/n
    for _ in range(iters):
        mrc = S.values @ w
        rc = w * mrc
        target = np.mean(rc)
        grad = rc - target
        # simple projected gradient
        w = w - 0.05 * grad
        w = _project_to_simplex_with_bounds(w, lb, ub)
    return pd.Series(w, index=px_hist.columns, dtype=float), "ERC"

# ---------- Black-Litterman (cost-aware) ----------
def black_litterman(px_hist: pd.DataFrame, base_cov: pd.DataFrame, mkt_w: pd.Series,
                    tau: float, P: np.ndarray, Q: np.ndarray, omega: Optional[np.ndarray]=None) -> Tuple[pd.Series, pd.DataFrame]:
    S = base_cov.values; w = mkt_w.values.reshape(-1,1)
    Pi = (S @ w).ravel()
    if omega is None: omega = np.diag(np.diag(P @ (tau*S) @ P.T))
    A = np.linalg.inv(tau*S)
    M = A + P.T @ np.linalg.inv(omega) @ P
    b = A @ Pi + P.T @ np.linalg.inv(omega) @ Q
    mu_bl = np.linalg.solve(M, b)
    mu_bl = pd.Series(mu_bl, index=px_hist.columns)
    cov_bl = pd.DataFrame(np.linalg.inv(M), index=px_hist.columns, columns=px_hist.columns)
    return mu_bl, cov_bl

def optimizer_black_litterman(px_hist: pd.DataFrame, lb: float, ub: float,
                              views_df: pd.DataFrame, tau: float=0.025, cost_bps: float=5.0) -> Tuple[pd.Series, str]:
    rets = to_returns(px_hist); S = ewma_cov(rets)
    w_mkt = inverse_variance_weights(px_hist, 0, 1.0)
    cols = list(px_hist.columns)
    P_list, Q_list = [], []
    for _, r in views_df.iterrows():
        asset = r.get("Asset",""); vtype = r.get("Type","abs"); val = float(r.get("Value",0.0))
        if asset not in cols: continue
        row = np.zeros(len(cols)); j = cols.index(asset)
        if vtype == "abs":
            row[j] = 1.0
        elif vtype == "relMkt":
            row[j] = 1.0; others = [i for i in range(len(cols)) if i != j]
            row[others] = -w_mkt.values[others] / max(1e-12, w_mkt.drop(asset).sum())
        else:
            row[j] = 1.0; others = [i for i in range(len(cols)) if i != j]; row[others] = -1.0/len(others)
        P_list.append(row); Q_list.append(val)
    if not P_list: return optimizer_min_vol(px_hist, lb, ub, use_ewma=True)
    P = np.vstack(P_list); Q = np.array(Q_list)
    mu_bl, cov_bl = black_litterman(px_hist, S, w_mkt, tau=tau, P=P, Q=Q)
    try:
        from pypfopt.efficient_frontier import EfficientFrontier
        ef = EfficientFrontier(mu_bl, cov_bl, weight_bounds=(lb, ub))
        w = pd.Series(ef.max_sharpe()); w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0); w = w / w.sum()
    except Exception:
        inv = 1.0 / (np.diag(cov_bl) + 1e-9)
        pref = inv * (mu_bl.values - mu_bl.values.mean())
        w = pd.Series(_project_to_simplex_with_bounds(pref, lb, ub), index=px_hist.columns)
    gamma = cost_bps/10000.0
    w = (1-gamma)*w + gamma*w_mkt.reindex_like(w).fillna(0.0); w = w / w.sum()
    return w, "Black-Litterman (cost-aware)"

# ---------- Caps & Risk Budgeting ----------
def apply_caps(weights: pd.Series, crypto_cap: float, equity_cap: float, single_cap: float) -> pd.Series:
    w = weights.copy().fillna(0.0).astype(float)
    if single_cap < 1.0:
        w = w.clip(upper=single_cap)
        if w.sum() > 0: w /= w.sum()
    crypto = [c for c in w.index if c in CRYPTOS]
    if crypto:
        cs = float(w.loc[crypto].clip(lower=0).sum())
        if crypto_cap <= 0:
            w.loc[crypto] = 0.0
            if w.sum() > 0: w /= w.sum()
        elif cs > crypto_cap and cs > 0:
            scale = crypto_cap / cs
            w.loc[crypto] = w.loc[crypto] * scale
            if w.sum() > 0: w /= w.sum()
    eq = [c for c in w.index if c in EQUITIES]
    if eq and equity_cap < 1.0:
        es = float(w.loc[eq].clip(lower=0).sum())
        if es > equity_cap and es > 0:
            scale = equity_cap / es
            w.loc[eq] = w.loc[eq] * scale
            if w.sum() > 0: w /= w.sum()
    return w

def enforce_risk_budget(weights: pd.Series, returns: pd.DataFrame, group_caps: Dict[str,float]) -> pd.Series:
    """Approximate risk-budgeting via scaling to keep group risk contributions under caps."""
    if weights.sum() == 0 or returns.empty: return weights
    S = returns.cov().reindex(index=weights.index, columns=weights.index).fillna(0.0)
    w = weights.copy().values
    names = list(weights.index)
    groups = {
        "Crypto": [i for i,n in enumerate(names) if names[i] in CRYPTOS],
        "Equity": [i for i,n in enumerate(names) if names[i] in EQUITIES],
    }
    for _ in range(20):
        MRC = S.values @ w
        RC = w * MRC
        tot = (w @ MRC) + 1e-12
        grp_rc = {}
        for g,idxs in groups.items():
            if not idxs: continue
            grp_rc[g] = float(RC[idxs].sum()/tot)
        scaled = False
        for g, cap in group_caps.items():
            if g in grp_rc and grp_rc[g] > cap + 1e-6:
                factor = cap / max(1e-12, grp_rc[g])
                # scale down all w in group slightly
                w[groups[g]] *= factor
                scaled = True
        if not scaled: break
        # renormalize
        w = np.clip(w, 0, None); 
        if w.sum()>0: w /= w.sum()
    return pd.Series(w, index=weights.index)

# ---------- UI Header ----------
colH1, colH2 = st.columns([0.8, 0.2])
with colH1:
    st.title("ALADDIN 3.2 — MAXI HEDGEFUND")
    st.caption(datetime.now().strftime("%d.%m.%Y %H:%M"))
with colH2:
    st.markdown("<div class='section' style='text-align:center;'>🌐<br/>Neon Mode</div>", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("Settings")
colA, colB = st.sidebar.columns([1,1])
with colA:
    if st.button("Clear data cache", use_container_width=True):
        st.cache_data.clear(); st.success("Cache cleared.")
with colB:
    preset_toggle = st.checkbox("Use KAYEN preset", value=True)

DEFAULT_FRIENDLY = ["BTCUSD","ETHUSD","SOLUSD","Apple","Tesla","Gold","NVIDIA","NASDAQ","S&P 500","MSCI World ETF","STARLINK (SPACE X)"]
friendly_selection = st.sidebar.multiselect(
    "Universe (choose from your set)", options=FRIENDLY_ORDER,
    default=DEFAULT_FRIENDLY if preset_toggle else DEFAULT_FRIENDLY
)
tickers = [ASSET_MAP[x] for x in friendly_selection]

years = st.sidebar.slider("Years of history", 2, 15, value=5)
outlier_thr = st.sidebar.slider("Outlier clamp (1d move)", 0.20, 0.80, 0.40, 0.05)
min_w = st.sidebar.slider("Min weight per asset", 0.00, 0.20, 0.00, step=0.01)
max_w = st.sidebar.slider("Max weight per asset", 0.10, 0.70, 0.35, step=0.01)

# Caps
single_cap = st.sidebar.slider("Single-asset cap", 0.10, 1.00, 0.45, step=0.05)
equity_cap = st.sidebar.slider("Equity cap (group)", 0.10, 1.00, 0.75, step=0.05)
crypto_cap = st.sidebar.slider("Crypto cap (group)", 0.00, 0.80, 0.20, step=0.01)

# Risk Budgeting caps (on risk contributions)
st.sidebar.markdown("**Risk Budgeting (RC caps)**")
rb_crypto = st.sidebar.slider("Crypto RC cap", 0.00, 0.80, 0.35, step=0.05)
rb_equity = st.sidebar.slider("Equity RC cap", 0.10, 1.00, 0.80, step=0.05)

alpha = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.90, 0.99, 0.95, step=0.01)
allow_short = st.sidebar.checkbox("Allow short (experimental)", value=False)
portfolio_value = st.sidebar.number_input("Portfolio value (BASE ccy)", 1000.0, 1e9, 100000.0, step=1000.0, format="%.2f")

# FX base + hedge
st.sidebar.markdown("**FX Base & Hedge**")
base_ccy = st.sidebar.selectbox("Base currency", ["EUR","USD","CHF","GBP"], index=0)
fx_hedge_ratio = st.sidebar.slider("FX hedge ratio (USD exposure)", 0.0, 1.0, 0.50, 0.05)

# Target-Vol + Leverage + Drawdown Guard
target_vol = st.sidebar.slider("Target annualized vol", 0.05, 0.40, 0.12, step=0.01)
max_leverage = st.sidebar.slider("Max gross leverage", 1.0, 3.0, 1.6, step=0.1)
dd_guard_thr = st.sidebar.slider("Drawdown guard threshold", 0.05, 0.30, 0.10, step=0.01)
dd_guard_strength = st.sidebar.slider("Guard strength", 0.0, 1.0, 0.50, step=0.05)
hedge_overlay_on = st.sidebar.checkbox("Enable Hedge Overlay (GLD + FX)", value=True)

# Momentum Tilt
mom_strength = st.sidebar.slider("Momentum tilt strength", 0.0, 1.0, 0.30, step=0.05)
mom_lb = st.sidebar.selectbox("Momentum lookback", ["6M","9M","12M"], index=2)

# Optimizer choice (single) + Ensemble blend
opt_choice = st.sidebar.selectbox("Optimizer", ["Min-Vol", "HRP", "Min-CVaR", "Black-Litterman", "ERC"], index=0)
use_ewma = st.sidebar.checkbox("Use EWMA cov in high-vol regime (auto)", value=True)
use_ensemble = st.sidebar.checkbox("Use Ensemble (blend optimizers)", value=True)

# Walk-forward settings + TC/Liquidity + Almgren–Chriss
wf_reb = st.sidebar.selectbox("Backtest rebalance", ["Monthly","Weekly"], index=0)
wf_cost_bps = st.sidebar.number_input("Base TC (bps per turnover)", 0, 300, 10, step=5)
wf_impact_k = st.sidebar.slider("Impact coef (sqrt model)", 0.0, 50.0, 8.0, step=0.5)
ac_temp = st.sidebar.slider("Almgren–Chriss temp impact (bp per %ADV)", 0.0, 50.0, 6.0, 0.5)
ac_perm = st.sidebar.slider("Almgren–Chriss perm impact (bp per %ADV)", 0.0, 30.0, 3.0, 0.5)
ac_slices = st.sidebar.slider("Execution slices", 1, 12, 4)
wf_turnover_cap = st.sidebar.slider("Turnover cap per rebalance", 0.05, 1.0, 0.35, step=0.05)
wf_min_trade = st.sidebar.slider("Min trade per asset (Δw)", 0.000, 0.050, 0.005, step=0.005)
wf_adv_days = st.sidebar.slider("ADV window (days)", 10, 60, 20, step=5)
wf_max_pct_adv = st.sidebar.slider("Max notional per rebalance as %ADV", 1.0, 50.0, 10.0, step=1.0)

# Monte Carlo settings
st.sidebar.markdown("**Monte Carlo**")
mc_block_bootstrap = st.sidebar.checkbox("Block-bootstrap instead of Gaussian", value=True)
mc_block = st.sidebar.slider("Bootstrap block length (days)", 3, 60, 10)
mc_horizon = st.sidebar.selectbox("VaR horizon", ["1d","5d","10d"], index=0)
mc_sims = st.sidebar.slider("MC Simulations", 1000, 30000, 8000, step=1000)

# Weights I/O
st.sidebar.markdown("**Weights I/O**")
upload_w = st.sidebar.file_uploader("Import weights.json", type=["json"])
if upload_w:
    try:
        wj = json.load(upload_w)
        st.session_state["import_weights"] = pd.Series(wj, dtype=float)
        st.sidebar.success("Weights importiert.")
    except Exception as e:
        st.sidebar.error(f"Weights-Import Fehler: {e}")

# Presets save/load
st.sidebar.markdown("**Presets**")
if st.sidebar.button("Save current preset"):
    preset = {
        "friendly_selection": friendly_selection, "years": years,
        "min_w": min_w, "max_w": max_w,
        "single_cap": single_cap, "equity_cap": equity_cap, "crypto_cap": crypto_cap,
        "rb_crypto": rb_crypto, "rb_equity": rb_equity,
        "alpha": alpha, "allow_short": allow_short,
        "portfolio_value": portfolio_value, "opt_choice": opt_choice, "use_ensemble": use_ensemble,
        "use_ewma": use_ewma, "wf_reb": wf_reb, "wf_cost_bps": wf_cost_bps,
        "wf_turnover_cap": wf_turnover_cap, "wf_min_trade": wf_min_trade,
        "target_vol": target_vol, "max_leverage": max_leverage,
        "dd_guard_thr": dd_guard_thr, "dd_guard_strength": dd_guard_strength, "hedge_overlay_on": hedge_overlay_on,
        "mom_strength": mom_strength, "mom_lb": mom_lb,
        "wf_adv_days": wf_adv_days, "wf_max_pct_adv": wf_max_pct_adv, "wf_impact_k": wf_impact_k,
        "ac_temp": ac_temp, "ac_perm": ac_perm, "ac_slices": ac_slices,
        "outlier_thr": outlier_thr,
        "mc_block_bootstrap": mc_block_bootstrap, "mc_block": mc_block, "mc_horizon": mc_horizon,
        "mc_sims": mc_sims,
        "base_ccy": base_ccy, "fx_hedge_ratio": fx_hedge_ratio
    }
    st.sidebar.download_button("Download preset.json", data=json.dumps(preset, indent=2).encode(),
                               file_name="preset.json", mime="application/json")
uploaded = st.sidebar.file_uploader("Load preset.json", type=["json"], key="preset_upl")
if uploaded and "preset_loaded_once" not in st.session_state:
    try:
        preset = json.load(uploaded); st.session_state.update(preset)
        st.sidebar.success("Preset loaded. Bitte Seite ggf. neu laden (R).")
        st.session_state["preset_loaded_once"] = True
    except Exception as e:
        st.sidebar.error(f"Preset error: {e}")

# ---------- Data ----------
if len(tickers) == 0:
    st.warning("Bitte mindestens einen Ticker wählen."); st.stop()

# Load asset prices
try:
    px_usd, vol = load_ohlcv(tickers, years=years, clamp_thr=outlier_thr)
    if px_usd.empty: st.error("Keine Preisdaten geladen — prüfe Ticker."); st.stop()
    rets_usd = to_returns(px_usd).dropna(how="all")
except Exception as e:
    st.error(f"Datenfehler: {e}"); st.stop()

# FX conversion to base currency
def fx_series(base_ccy: str, index_like: pd.DatetimeIndex) -> pd.Series:
    if base_ccy == "USD" or FX_BASES[base_ccy] is None:
        return pd.Series(1.0, index=index_like, name="FX")
    sym = FX_BASES[base_ccy]
    fx_px = load_prices([sym], years=years, clamp_thr=0.25)
    if sym in fx_px.columns: s = fx_px[sym].reindex(index_like).ffill().bfill()
    else: s = pd.Series(1.0, index=index_like)
    return s.rename("FX")

# For USD-quoted assets, base wealth in chosen base_ccy
fx = fx_series(base_ccy, px_usd.index)
# USD asset price converted into base_ccy: if base=EUR and sym=EURUSD (USD per EUR), USD_px / EURUSD -> base EUR
px_base = px_usd.div(fx, axis=0)
rets_base = to_returns(px_base).dropna(how="all")

# FX hedge return series (hedging USD exposure back to base)
fx_ret = to_returns(fx.to_frame("FX")).squeeze() * -1.0  # short USD when base!=USD

# Data Quality quick checks
dq_missing = float(px_base.isna().mean().mean())
dq_min_hist = int(px_base.notna().sum().min())
dq_last = px_base.index.max()
dq_last_str = dq_last.date().isoformat() if hasattr(dq_last, "date") else str(dq_last)
if dq_missing > 0.02 or dq_min_hist < 250:
    st.warning(f"Data Quality: missing={dq_missing:.1%}, min_history={dq_min_hist} Tage. Latest={dq_last_str}")

# ---------- Regime & Risk model ----------
regime = regime_tag(rets_base)
use_ewma_now = (use_ewma and regime in ["high-vol","bear"])
target_vol_regime_scale = 0.85 if regime in ["high-vol","bear"] else 1.0
target_vol_eff = max(0.03, target_vol * target_vol_regime_scale)

# ---------- Black-Litterman Views UI ----------
default_views = pd.DataFrame({"Asset":[tickers[0] if len(tickers) else ""], "Type":["abs"], "Value":[0.08]})
views_df = None
if opt_choice == "Black-Litterman" or use_ensemble:
    st.sidebar.markdown("**Black-Litterman Views**")
    st.sidebar.caption("Type: abs = exp(annual); rel = vs equal basket; relMkt = vs market weights")
    views_df = st.sidebar.data_editor(default_views, num_rows="dynamic", use_container_width=True)

# If imported weights provided, use as blend hint
imported = st.session_state.get("import_weights")
imp = None
if imported is not None:
    imp = imported.reindex(px_base.columns).fillna(0.0).clip(lower=0.0)
    if imp.sum() > 0: imp /= imp.sum()
    imp = apply_caps(imp, crypto_cap, equity_cap, single_cap)

# ---------- Core optimizations ----------
def run_single_optimizer(name:str, px_hist:pd.DataFrame) -> pd.Series:
    if name == "Min-Vol":
        return optimizer_min_vol(px_hist, lb=min_w, ub=max_w, use_ewma=use_ewma_now)[0]
    if name == "HRP":
        return optimizer_hrp(px_hist, lb=min_w, ub=max_w)[0]
    if name == "Min-CVaR":
        return optimizer_cvar(to_returns(px_hist), alpha=alpha, lb=min_w, ub=max_w, allow_short=allow_short)[0]
    if name == "Black-Litterman":
        try:
            return optimizer_black_litterman(px_hist, lb=min_w, ub=max_w,
                    views_df=views_df if views_df is not None else pd.DataFrame(columns=["Asset","Type","Value"]))[0]
        except Exception:
            return optimizer_min_vol(px_hist, lb=min_w, ub=max_w, use_ewma=True)[0]
    if name == "ERC":
        return optimizer_erc(px_hist, lb=min_w, ub=max_w, use_ewma=use_ewma_now)[0]
    # fallback
    return inverse_variance_weights(px_hist, min_w, max_w)

OPT_LIST = ["Min-Vol","HRP","Min-CVaR","Black-Litterman","ERC"]

def ensemble_blend(px_hist: pd.DataFrame, returns: pd.DataFrame) -> Tuple[pd.Series, Dict[str,float]]:
    """CV-basiertes Blending: Finde Nichtnegativ-Gewichte der Optimizer, die OOS-Sharpe maximieren."""
    models = {}
    for o in OPT_LIST:
        try:
            w = run_single_optimizer(o, px_hist).reindex(px_hist.columns).fillna(0.0)
            if w.sum()>0: w/=w.sum()
            models[o] = w
        except Exception:
            continue
    if not models:
        return inverse_variance_weights(px_hist, min_w, max_w), { "IV": 1.0 }

    # Purged K-Fold
    k, embargo = 5, 10
    idx = returns.index
    fold_size = len(idx)//k if k>0 else len(idx)
    X, y = [], []
    for i in range(k):
        start = i*fold_size; end = (i+1)*fold_size if i<k-1 else len(idx)
        te_idx = idx[start:end]
        pre_end = max(0, start-embargo); post_start = min(len(idx), end+embargo)
        tr_idx = idx[:pre_end].append(idx[post_start:])
        # Compute each model OOS Sharpe on te
        feats = []
        for o,w in models.items():
            r_te = returns.loc[te_idx].reindex(columns=w.index).fillna(0.0).dot(w.values)
            sr_te = sharpe_ratio(r_te)
            feats.append(sr_te)
        X.append(feats); y.append(1.0) # dummy target
    if not X:  # on very short history
        avg = sum(models.values())/len(models)
        return avg/avg.sum(), {o: 1/len(models) for o in models}

    X = np.array(X)
    # Nonnegative weights sum to 1 -> simple uniform init + project
    blend = np.ones(X.shape[1]) / X.shape[1]
    # Score = mean OOS Sharpe of blended signal
    for _ in range(200):
        grad = np.mean(X, axis=0)  # gradient approx
        blend = blend + 0.1*(grad - np.mean(grad))
        blend = np.clip(blend, 0, None)
        if blend.sum()>0: blend/=blend.sum()
    # combine model weights
    w_sum = None
    for i,(o,w) in enumerate(models.items()):
        part = blend[i] * w
        w_sum = part if w_sum is None else w_sum.add(part, fill_value=0.0)
    w_sum = w_sum.fillna(0.0)
    if w_sum.sum()>0: w_sum/=w_sum.sum()
    blend_map = {o: float(blend[i]) for i,o in enumerate(models.keys())}
    return w_sum, blend_map

# Select single or ensemble
if use_ensemble:
    w_opt_raw, blend_map = ensemble_blend(px_base, rets_base)
    opt_label = "Ensemble: " + ", ".join([f"{k}:{v:.0%}" for k,v in blend_map.items()])
else:
    w_opt_raw = run_single_optimizer(opt_choice, px_base).reindex(px_base.columns).fillna(0.0)
    if w_opt_raw.sum()>0: w_opt_raw/=w_opt_raw.sum()
    opt_label = opt_choice

# Optional import blend (user)
if imp is not None and imp.sum()>0:
    w_opt_raw = (0.7*w_opt_raw + 0.3*imp)
    if w_opt_raw.sum()>0: w_opt_raw/=w_opt_raw.sum()

# Caps + Risk budgeting
w_capped = apply_caps(w_opt_raw, crypto_cap, equity_cap, single_cap)
if w_capped.sum()>0: w_capped/=w_capped.sum()
w_opt = enforce_risk_budget(w_capped, rets_base, {"Crypto": rb_crypto, "Equity": rb_equity})
if w_opt.sum()>0: w_opt/=w_opt.sum()

# Momentum Tilt
lb_map = {"6M":126, "9M":189, "12M":252}; lb = lb_map[mom_lb]
mom = (px_base/px_base.shift(lb) - 1.0).iloc[-1].reindex(w_opt.index).fillna(0.0)
mom_score = (mom.rank(pct=True) - 0.5)
w_tilt = w_opt * (1 + mom_strength * mom_score)
if w_tilt.sum() > 0: w_opt = (w_tilt / w_tilt.sum()).fillna(0.0)

# ---------- Portfolio series (base ccy) & FX-hedge ----------
port_core = (rets_base.fillna(0.0) @ w_opt.values).rename("Portfolio")
# Add FX hedge overlay (hedge USD back to base)
if base_ccy != "USD" and fx_hedge_ratio > 0:
    port_core = port_core + fx_hedge_ratio * fx_ret.reindex(port_core.index).fillna(0.0)

# Hedge overlay (GLD + extra FX) when enabled & regime/bad DD
cum_core = (1.0 + port_core).cumprod()
dd_live = (cum_core/cum_core.cummax()-1.0).iloc[-1]
hedge_add = 0.0
if hedge_overlay_on:
    if dd_live < -dd_guard_thr or regime in ["bear","high-vol"]:
        hedge_add = min(0.10, dd_guard_strength*0.10)  # up to 10% overlay
port = port_core + hedge_add * rets_base.get("GLD", pd.Series(0.0, index=port_core.index)).reindex(port_core.index).fillna(0.0)

cum  = (1.0 + port).cumprod()
ann_ret, ann_vol = annualize_stats(port)
sr = sharpe_ratio(port); VaR, ES = hist_var_es(port, alpha=alpha)
MDD = max_drawdown(cum); skew, kurt = higher_moments(port)

# Risk targeting + Drawdown Guard (adaptive leverage)
eps = 1e-8
lev_base = min(max_leverage, (target_vol_eff / max(eps, ann_vol)))
guard_scale = 1.0
if dd_live < -dd_guard_thr:
    over = abs(dd_live) - dd_guard_thr
    guard_scale = max(0.3, 1.0 - dd_guard_strength * (over / max(1e-6, dd_guard_thr)))
scale_live = max(0.1, lev_base * guard_scale)

port_eff = port * scale_live
cum_eff  = (1.0 + port_eff).cumprod()
ann_ret_eff, ann_vol_eff = annualize_stats(port_eff)
sr_eff  = sharpe_ratio(port_eff); VaR_eff, ES_eff = hist_var_es(port_eff, alpha=alpha)
MDD_eff = max_drawdown(cum_eff)
roll_sharpe = port_eff.rolling(60).apply(lambda x: 0.0 if np.std(x)==0 else (np.mean(x)/np.std(x))*np.sqrt(252), raw=True)

# ---------- Tabs ----------
tab_overview, tab_risk, tab_backtest, tab_cv, tab_pbo, tab_factors, tab_stress, tab_trades, tab_broker, tab_ai, tab_report = st.tabs(
    ["Overview", "Risk", "Backtest", "CV & DS", "Reality Check (PBO)", "Factors", "Stress", "Trades", "Paper Broker", "AI Insights", "Report"]
)

# ---------- Overview ----------
def kpi(label, value):
    st.markdown(f"""
    <div class="kpi-card"><div class="kpi-label">{label}</div>
      <div class="kpi-big">{value}</div>
    </div>""", unsafe_allow_html=True)

with tab_overview:
    a,b,c,d = st.columns(4)
    with a: kpi("Portfolio Index", f"{cum_eff.iloc[-1]:.2f}x")
    with b: kpi("Sharpe (post-scale)", f"{sr_eff:.2f}")
    with c: kpi(f"VaR({int(alpha*100)}%) 1d", f"{VaR_eff:.2%}")
    with d: kpi("Max Drawdown", f"{MDD_eff:.2%}")
    st.caption(f"Model: {opt_label} · Caps SA {single_cap:.0%} / Eq {equity_cap:.0%} / Crypto {crypto_cap:.0%} | Regime: {regime} · Lev: {scale_live:.2f}× · Base: {base_ccy} · FXH: {fx_hedge_ratio:.0%} · HedgeAdd: {hedge_add:.0%}")

    sa, sb, sc, sd = st.columns(4)
    with sa: st.plotly_chart(sparkline((1+port_eff).cumprod()), use_container_width=True, config={"displayModeBar": False})
    with sb: st.plotly_chart(sparkline(roll_sharpe.fillna(0)), use_container_width=True, config={"displayModeBar": False})
    with sc: st.plotly_chart(sparkline((cum_eff/cum_eff.cummax()-1.0).fillna(0)), use_container_width=True, config={"displayModeBar": False})
    with sd: st.plotly_chart(sparkline(rets_base.mean(axis=1).rolling(20).mean().fillna(0)), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    L, R = st.columns([1.25, 1.0])
    fr_names = {ASSET_MAP[k]:k for k in ASSET_MAP}
    with L:
        st.subheader("Portfolio Overview (target)")
        ytd = {t: rets_base[t].loc[rets_base.index.year == rets_base.index[-1].year].sum() if t in rets_base.columns else 0.0 for t in px_base.columns}
        df_over = pd.DataFrame({"Weight": w_opt.round(4), "YTD Return": pd.Series(ytd).round(4)})
        df_over.index = [fr_names.get(idx, idx) for idx in df_over.index]
        st.dataframe(df_over, use_container_width=True)
        st.download_button("Export Weights (JSON)",
            data=json.dumps({k: float(v) for k,v in w_opt.round(6).to_dict().items()}, indent=2).encode(),
            file_name="weights.json", mime="application/json")
    with R:
        st.subheader("Risk Metrics (post-scale)")
        df_risk = pd.DataFrame({
            "Annualized Return": [ann_ret_eff],
            "Annualized Volatility": [ann_vol_eff],
            "Sharpe": [sr_eff], "Skew": [skew], "Kurtosis (excess)": [kurt],
            f"VaR({int(alpha*100)}%)": [VaR_eff], f"CVaR({int(alpha*100)}%)~": [ES_eff],
            "Max Drawdown": [MDD_eff], "Leverage (×)": [scale_live],
        }).round(4)
        st.dataframe(df_risk, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Correlation Heatmap (base ccy)")
    fig, ax = plt.subplots()
    corr = rets_base.corr()
    im = ax.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels([fr_names.get(c,c) for c in corr.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels([fr_names.get(c,c) for c in corr.index])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    st.pyplot(fig)

# ---------- Risk (MC VaR/ES) ----------
def _gaussian_mc(rets: pd.DataFrame, w: pd.Series, scale: float, sims: int, horizon_d: int):
    mu = rets.mean().values; S  = rets.cov().values
    R = np.random.multivariate_normal(mu, S, size=(sims, horizon_d))
    port_h = (R @ w.values).sum(axis=1) * scale
    return port_h

def _block_bootstrap_mc(rets: pd.DataFrame, w: pd.Series, scale: float, sims: int, horizon_d: int, block_len: int):
    r = rets.values; T, N = r.shape; blocks = []
    starts = np.random.randint(0, max(1, T-block_len), size=(sims, math.ceil(horizon_d/block_len)))
    for s in range(sims):
        seq = []
        for idx in starts[s]:
            b = r[idx:idx+block_len]
            if len(b) < block_len:
                pad = r[:(block_len-len(b))]; b = np.vstack([b, pad])
            seq.append(b)
        path = np.vstack(seq)[:horizon_d]; blocks.append(path)
    R = np.stack(blocks, axis=0)
    port_h = (R @ w.values[:,None]).sum(axis=1).ravel() * scale
    return port_h

with tab_risk:
    st.subheader("Drawdown & Rolling Sharpe (post-scale)")
    fig1 = go.Figure()
    dd = (cum_eff/cum_eff.cummax()-1.0)
    fig1.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines"))
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CFG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="Rolling Sharpe (60d)", mode="lines"))
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)

    st.subheader("Monte Carlo VaR/ES (post-scale)")
    H = {"1d":1, "5d":5, "10d":10}[mc_horizon]
    try:
        sims = _block_bootstrap_mc(rets_base, w_opt, scale_live, mc_sims, H, mc_block) if mc_block_bootstrap \
               else _gaussian_mc(rets_base, w_opt, scale_live, mc_sims, H)
        VaR_mc = np.percentile(sims, (1-alpha)*100)
        ES_mc  = sims[sims<=VaR_mc].mean() if np.any(sims<=VaR_mc) else VaR_mc
        st.write(f"MC VaR({int(alpha*100)}%) {mc_horizon}: **{VaR_mc:.2%}**,  ES: **{ES_mc:.2%}**  ·  Sims: {mc_sims:,}  ({'Bootstrap' if mc_block_bootstrap else 'Gaussian'})")
    except Exception as e:
        st.warning(f"Monte Carlo nicht verfügbar: {e}")

# ---------- Backtest (Walk-Forward, AC model) ----------
def monthly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    g = pd.Series(index=index, data=True); return g.groupby([index.year, index.month]).head(1).index
def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    g = pd.Series(index=index, data=True); return g.groupby([index.year, index.isocalendar().week]).head(1).index

def almgren_chriss_cost(trade_frac: pd.Series, eq: float, adv_notional: pd.Series, slices:int, ac_temp:float, ac_perm:float) -> float:
    """
    trade_frac: absolute Δw (per asset) executed this rebalance
    eq: current equity (index units)
    adv_notional: %ADV capacity (notional)
    temp cost ~ ac_temp * sum( (notional/ADV) / slices )^0.5
    perm cost ~ ac_perm * sum( (notional/ADV) )
    """
    if trade_frac.sum()<=0: return 0.0
    notional = trade_frac * eq
    cap = adv_notional.replace(0, np.nan)
    ratio = (notional / cap).clip(lower=0.0)
    temp = (ac_temp/10000.0) * np.nansum(np.sqrt(ratio) / max(1, slices))
    perm = (ac_perm/10000.0) * np.nansum(ratio)
    return float(temp + perm)

def rebalance_backtest(px: pd.DataFrame, rets: pd.DataFrame, vol_df: pd.DataFrame, dates: pd.DatetimeIndex,
                       lb: float, ub: float, single_cap: float, equity_cap: float, crypto_cap: float,
                       base_bps: float, impact_k: float, adv_days: int, max_pct_adv: float,
                       turnover_cap: float, min_trade: float,
                       opt_choice: str, use_ewma: bool,
                       alpha: float, allow_short: bool, views_df: Optional[pd.DataFrame],
                       target_vol: float, max_leverage: float, mom_strength: float, mom_lb_days: int,
                       base_ccy:str, fx_ret: pd.Series,
                       dd_thr: float, dd_strength: float, hedge_overlay_on: bool,
                       ac_temp:float, ac_perm:float, ac_slices:int,
                       min_hist: int=63) -> Tuple[pd.Series, pd.Series]:
    w_prev = None; eq_path, tc_series = [], []
    equity = 1.0
    adv = (vol_df.rolling(adv_days).mean() * px).fillna(0.0)  # notional ADV
    cum_equity = 1.0; cum_peak = 1.0
    for dt in rets.index:
        if dt in dates:
            px_hist = px.loc[:dt].dropna()
            if px_hist.shape[0] < min_hist:
                w = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)
            else:
                # ensemble inside WF: cheaper single for speed
                w = run_single_optimizer(opt_choice, px_hist)
                w = apply_caps(w, crypto_cap, equity_cap, single_cap)
                if w.sum() > 0: w = w / w.sum()
                mom = (px_hist/px_hist.shift(mom_lb_days) - 1.0).iloc[-1].reindex(w.index).fillna(0.0)
                mom_score = (mom.rank(pct=True) - 0.5)
                w_tilt = w * (1 + mom_strength * mom_score)
                if w_tilt.sum() > 0: w = (w_tilt / w_tilt.sum()).fillna(0.0)

            if w_prev is not None:
                delta = (w - w_prev).abs()
                delta = delta.where(delta >= min_trade, 0.0)
                adv_now = adv.reindex(index=[dt]).ffill().iloc[0].reindex(w.index).replace(0, np.nan)
                desired_notional = delta * equity
                max_notional = (max_pct_adv/100.0) * adv_now
                scale = 1.0
                mask = (~max_notional.isna()) & (max_notional>0)
                if mask.any():
                    ratio = (desired_notional[mask] / max_notional[mask]).max()
                    if ratio > 1.0: scale = 1.0/float(ratio)
                delta *= scale
                if float(delta.sum()) > turnover_cap:
                    delta = delta * (turnover_cap / float(delta.sum()))
                # Impact cost (legacy sqrt + AC light)
                traded_notional = delta * equity
                imp = pd.Series(0.0, index=w.index, dtype=float)
                if mask.any():
                    z = traded_notional[mask] / max_notional[mask].replace(0,np.nan)
                    imp.loc[mask] = (base_bps/10000.0) + (impact_k/10000.0) * np.sqrt(z.clip(lower=0.0).fillna(0.0))
                else:
                    imp += (base_bps/10000.0)
                tc_legacy = float(imp.fillna(base_bps/10000.0).sum())
                tc_ac = almgren_chriss_cost(delta, equity, max_notional, ac_slices, ac_temp, ac_perm)
                tc = tc_legacy + tc_ac
                equity *= (1.0 - tc); tc_series.append((dt, tc))
                w = w_prev + np.sign(w - w_prev) * delta
                w = w / w.sum()
            w_prev = w.copy()

        if w_prev is None:
            w_prev = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)

        # Risk targeting (63d)
        hist = rets.loc[:dt].tail(63).reindex(columns=w_prev.index).dot(w_prev.values)
        curr_vol = hist.std() * np.sqrt(252) if len(hist) else 0.0
        sf = min(max_leverage, (target_vol / max(1e-8, curr_vol))) if curr_vol > 0 else 1.0

        # Drawdown guard & hedge overlay
        cum_peak = max(cum_peak, equity)
        dd = equity/cum_peak - 1.0
        guard = 1.0
        if dd < -dd_thr:
            over = abs(dd) - dd_thr
            guard = max(0.3, 1.0 - dd_strength * (over/max(1e-6, dd_thr)))
        r_core = float(rets.loc[dt].reindex(w_prev.index).fillna(0.0).dot(w_prev.values))
        r = r_core
        if base_ccy != "USD": r += fx_ret.reindex([dt]).fillna(0.0).iloc[0] * 0.0  # already handled ex-ante in px

        # Hedge overlay GLD when needed
        if hedge_overlay_on and (dd < -dd_thr):
            r += 0.05 * float(rets.get("GLD", pd.Series(0.0, index=[dt])).reindex([dt]).fillna(0.0).iloc[0])

        equity *= (1.0 + guard * sf * r)
        eq_path.append(equity)

    tc_series = pd.Series({d: v for d,v in tc_series}).reindex(rets.index).fillna(0.0).rename("TC")
    return pd.Series(eq_path, index=rets.index, name="Portfolio"), tc_series

with tab_backtest:
    st.subheader("Walk-Forward Backtest · Portfolio vs Benchmark (TC/Liquidity-aware + AC)")
    bt_years = st.slider("Backtest years", 2, 15, value=min(5, years), key="bt_years")
    px_bt, vol_bt = load_ohlcv(tickers, years=bt_years, clamp_thr=outlier_thr)
    # convert to base
    fx_bt = fx_series(base_ccy, px_bt.index)
    px_bt_base = px_bt.div(fx_bt, axis=0)
    rets_bt = to_returns(px_bt_base).dropna()
    rebal_dates = monthly_dates(rets_bt.index) if wf_reb == "Monthly" else weekly_dates(rets_bt.index)

    eq, tc_series = rebalance_backtest(
        px_bt_base, rets_bt, vol_bt, rebal_dates,
        lb=min_w, ub=max_w,
        single_cap=single_cap, equity_cap=equity_cap, crypto_cap=crypto_cap,
        base_bps=wf_cost_bps, impact_k=wf_impact_k, adv_days=wf_adv_days,
        max_pct_adv=wf_max_pct_adv, turnover_cap=wf_turnover_cap,
        min_trade=wf_min_trade,
        opt_choice=opt_choice, use_ewma=use_ewma_now, alpha=alpha, allow_short=allow_short,
        views_df=views_df, target_vol=target_vol_eff, max_leverage=max_leverage,
        mom_strength=mom_strength, mom_lb_days=lb,
        base_ccy=base_ccy, fx_ret=to_returns(fx_bt.to_frame("FX")).squeeze()*-1.0,
        dd_thr=dd_guard_thr, dd_strength=dd_guard_strength, hedge_overlay_on=hedge_overlay_on,
        ac_temp=ac_temp, ac_perm=ac_perm, ac_slices=ac_slices
    )

   # -------- Benchmark robust laden & in Basiswährung bringen --------
bench_ticker = st.selectbox("Benchmark", ["SPY", "QQQ", "URTH", "GLD"], index=0, key="bench")

def _force_series(obj: pd.DataFrame | pd.Series, prefer_col: str | None = None) -> pd.Series:
    """Nimmt DataFrame oder Series entgegen und gibt IMMER eine 1D-Series zurück."""
    if isinstance(obj, pd.DataFrame):
        if prefer_col is not None and prefer_col in obj.columns:
            s = obj[prefer_col]
        else:
            s = obj.iloc[:, 0]
        return pd.Series(s).dropna()
    return pd.Series(obj).dropna()

try:
    bench_raw = _download_with_retry(bench_ticker, period=f"{bt_years}y")
    bench_df  = _normalize_price_frame(bench_raw)
    bench_usd = _force_series(bench_df, prefer_col=bench_ticker)
except Exception:
    try:
        bench_usd = yf.download(bench_ticker, period=f"{bt_years}y",
                                auto_adjust=True, progress=False)["Close"].dropna()
    except Exception:
        bench_usd = pd.Series(dtype=float)

# in Basiswährung umrechnen
fx_bench = fx_series(base_ccy, bench_usd.index)
bench_px  = bench_usd / fx_bench

# -------- Gemeinsame Timeline & Performance-Kurven --------
common = eq.index.intersection(bench_px.index)

if len(common) >= 2:
    port_eq  = (eq.loc[common] / float(eq.loc[common].iloc[0])).astype(float)
    bench_eq = (bench_px.loc[common] / float(bench_px.loc[common].iloc[0])).astype(float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=common, y=port_eq.values,  name="Portfolio", mode="lines"))
    fig.add_trace(go.Scatter(x=common, y=bench_eq.values, name=bench_ticker, mode="lines"))
    fig.update_layout(title="Index (Start=1.0)", xaxis_title="Date", yaxis_title="Index",
                      height=420, template="aladdin")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    bt_ret = eq.pct_change().dropna()
    n = len(bt_ret) if len(bt_ret) else 1
    bt_cagr = (eq.iloc[-1] ** (252/n)) - 1 if len(eq) else 0.0
    bt_vol  = bt_ret.std() * np.sqrt(252) if len(bt_ret) else 0.0
    bt_sha  = (bt_ret.mean()/bt_ret.std()) * np.sqrt(252) if len(bt_ret) and bt_ret.std()!=0 else 0.0
    bt_mdd  = float((eq/eq.cummax() - 1.0).min()) if len(eq) else 0.0
    tc_ann  = float(tc_series.mean()*252) if len(tc_series) else 0.0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Backtest CAGR", f"{bt_cagr:.2%}")
    c2.metric("Backtest Vol", f"{bt_vol:.2%}")
    c3.metric("Backtest Sharpe", f"{bt_sha:.2f}")
    c4.metric("Backtest MaxDD", f"{bt_mdd:.2%}")
    c5.metric("Avg TC (ann.)", f"{tc_ann:.2%}")

    last_port  = float(port_eq.iloc[-1])
    last_bench = float(bench_eq.iloc[-1])
    outp = last_port / last_bench - 1.0
    st.markdown(f"**Outperformance vs {bench_ticker}: {outp:.2%}**")
else:
    st.info("Benchmark-Zeitreihe zu kurz oder keine Überschneidung mit Portfolio-History.")

# ---------- CV & Deflated Sharpe ----------
def time_series_folds(index: pd.DatetimeIndex, k: int=5, embargo: int=10):
    n = len(index); fold_size = n // k
    for i in range(k):
        start = i*fold_size; end = (i+1)*fold_size if i<k-1 else n
        test_idx = index[start:end]
        pre_end = max(0, start-embargo); post_start = min(n, end+embargo)
        train_idx = index[:pre_end].append(index[post_start:])
        yield train_idx, test_idx

def deflated_sharpe(sr_hat: float, T: int, sr_mean: float, sr_std: float, n_trials: int) -> float:
    if n_trials <= 1 or sr_std == 0: return 0.0
    emax = sr_mean + sr_std * (1 - 0.75*(np.log(np.log(n_trials))) / np.sqrt(2*np.log(n_trials)))
    dsr = (sr_hat - emax) * np.sqrt(max(1, T))
    return float(dsr)

with tab_cv:
    st.subheader("Purged K-Fold Cross-Validation & Deflated Sharpe")
    k = st.slider("Folds (K)", 3, 10, 5)
    embargo = st.slider("Embargo (days)", 0, 30, 10)
    if len(port_eff) > 200:
        srs = []
        for tr_idx, te_idx in time_series_folds(port_eff.index, k=k, embargo=embargo):
            te = port_eff.reindex(te_idx).dropna()
            sr_te = sharpe_ratio(te) if len(te)>0 else 0.0
            srs.append(sr_te)
        cv_sr_mean = float(np.mean(srs)) if srs else 0.0
        cv_sr_std  = float(np.std(srs)) if srs else 0.0
        T = len(port_eff.dropna())
        ds = deflated_sharpe(sr_eff, T, cv_sr_mean, cv_sr_std, n_trials=k)
        c1,c2,c3 = st.columns(3)
        c1.metric("CV Sharpe (mean)", f"{cv_sr_mean:.2f}")
        c2.metric("CV Sharpe (std)", f"{cv_sr_std:.2f}")
        c3.metric("Deflated Sharpe", f"{ds:.2f}")
        st.caption("Deflated Sharpe ~ Overfitting-korrigierter Sharpe (Bailey-Approx, grob).")
    else:
        st.info("Zu wenig History für CV. Erhöhe 'Years of history' oder nutze breiteres Universum.")

# ---------- Reality Check / PBO ----------
def pbo_probability(returns: pd.DataFrame, opt_names: List[str], trials:int=20) -> float:
    """Combinatorial CV surrogate: wähle IS-Best-Optimizer, prüfe OOS ob er schlechter als Median ist."""
    if returns.isna().any().any() or len(returns)<250: return np.nan
    pbo_hits = 0; total = 0
    idx = returns.index
    for _ in range(trials):
        cut = np.random.randint(int(0.3*len(idx)), int(0.7*len(idx)))
        is_idx = idx[:cut]; oos_idx = idx[cut:]
        srs_is = returns.loc[is_idx].apply(sharpe_ratio, axis=0)
        best = srs_is.idxmax()
        srs_oos = returns.loc[oos_idx].apply(sharpe_ratio, axis=0)
        if best in srs_oos.index:
            # underperform vs. median?
            if srs_oos[best] < srs_oos.median(): pbo_hits+=1
            total+=1
    return (pbo_hits/total) if total>0 else np.nan

with tab_pbo:
    st.subheader("Reality Check (PBO) — Probability of Backtest Overfitting")
    # Build strategy returns from optimizers on full sample (fixed weights over time; simplistic proxy)
    strat_rets = {}
    for o in OPT_LIST:
        try:
            w = run_single_optimizer(o, px_base)
            if w.sum()>0: w/=w.sum()
            r = rets_base.reindex(columns=w.index).fillna(0.0).dot(w.values)
            strat_rets[o] = r
        except Exception:
            continue
    if strat_rets:
        R = pd.DataFrame(strat_rets).dropna()
        pbo = pbo_probability(R, list(R.columns), trials=40)
        st.write(f"PBO (lower is better): **{pbo:.2%}**" if pd.notna(pbo) else "PBO: n/a")
        st.line_chart(R.rolling(60).apply(lambda x: np.mean(x)/np.std(x)*np.sqrt(252) if np.std(x)>0 else 0.0))
    else:
        st.info("Nicht genug Daten/Strategien für PBO.")

# ---------- Factors ----------
with tab_factors:
    st.subheader("Factor Attribution (Daily Returns Regression)")
    factor_map = {"Stocks (SPY)": "SPY", "Bonds (IEF)": "IEF", "Gold (GLD)": "GLD", "USD (UUP proxy)": "UUP"}
    fac_px_usd = load_prices(list(factor_map.values()), years=min(5, years), clamp_thr=outlier_thr)
    # convert to base ccy
    fac_px = fac_px_usd.div(fx, axis=0).dropna(how="all")
    fac_rets = to_returns(fac_px)

    y = port_eff.reindex(fac_rets.index).dropna()
    X = fac_rets.reindex(y.index).fillna(0.0).copy()
    X.columns = list(factor_map.keys())

    beta = {}; r2 = 0.0
    try:
        import statsmodels.api as sm
        X_ = sm.add_constant(X); model = sm.OLS(y.values, X_.values).fit()
        r2 = float(model.rsquared); coefs = dict(zip(["Const"] + list(X.columns), model.params))
        for kf in X.columns: beta[kf] = float(coefs.get(kf, 0.0))
    except Exception:
        X_ = np.column_stack([np.ones(len(X)), X.values])
        coef, *_ = np.linalg.lstsq(X_, y.values, rcond=None)
        pred = X_ @ coef
        ss_res = np.sum((y.values - pred)**2); ss_tot = np.sum((y.values - y.values.mean())**2)
        r2 = 0.0 if ss_tot==0 else 1 - ss_res/ss_tot
        for i, kf in enumerate(X.columns, start=1): beta[kf] = float(coef[i])

    betas_df = pd.DataFrame({"Beta": beta}).T.round(4)
    st.dataframe(betas_df, use_container_width=True)
    st.write(f"R² (explained variance): **{r2:.2f}**")

# ---------- Stress Testing ----------
with tab_stress:
    st.subheader("Instant & Historical Stress")
    st.caption("1-Tages-Schocks (instant) & historische Szenarien mapped auf das Portfolio (post-scale).")
    scenario_lib = {
        "Tech + Crypto Crash": {"AAPL": -0.15, "TSLA": -0.18, "QQQ": -0.12, "BTC-USD": -0.25, "ETH-USD": -0.35, "NVDA": -0.16},
        "Rates Spike": {"URTH": -0.05, "SPY": -0.04, "GLD": -0.03},
        "USD Surge": {"GLD": -0.03, "SPY": -0.03},
        "Crypto Winter": {"BTC-USD": -0.30, "ETH-USD": -0.40, "SOL-USD": -0.45},
        "2008 Crisis Day": {"SPY": -0.09, "QQQ": -0.08, "GLD": 0.02, "URTH": -0.08},
        "2020 Covid Crash Day": {"SPY": -0.12, "QQQ": -0.11, "GLD": -0.03, "URTH": -0.11},
        "2022 Inflation Shock": {"SPY": -0.04, "QQQ": -0.06, "GLD": -0.01, "UFO": -0.05},
        "Gold Spike": {"GLD": 0.05, "SPY": -0.02, "QQQ": -0.03},
    }
    scn = st.selectbox("Scenario", list(scenario_lib.keys()), index=0)
    pick = st.multiselect("Assets to shock (optional override)", options=list(px_base.columns), default=[])
    shock_pct = st.slider("Shock size (manual, %)", -50, 50, -5, step=1)

    left, right = st.columns(2)
    with left:
        if st.button("Apply Scenario"):
            shock = pd.Series(0.0, index=px_base.columns, dtype=float)
            for k, v in scenario_lib[scn].items():
                if k in shock.index: shock.loc[k] = v
            pnl = float((w_opt * shock).sum()) * float(scale_live)
            st.success(f"[{scn}] Estimated instant P&L (post-scale): **{pnl:.2%}**")
    with right:
        if st.button("Apply Manual Shock"):
            if not pick: st.info("Bitte mindestens ein Asset auswählen.")
            else:
                shock = pd.Series(0.0, index=px_base.columns, dtype=float)
                shock.loc[pick] = shock_pct / 100.0
                pnl = float((w_opt * shock).sum()) * float(scale_live)
                st.info(f"Manual shock P&L (post-scale): **{pnl:.2%}**")

# ---------- Trades / Rebalance Planner ----------
with tab_trades:
    st.subheader("Rebalance Planner")
    if "current_weights" not in st.session_state:
        st.session_state.current_weights = w_opt.copy()

    edited = st.data_editor(pd.DataFrame({"Current": st.session_state.current_weights.round(4)}),
                            use_container_width=True, key="cur_edit")
    current = edited["Current"].reindex(w_opt.index).fillna(0.0)

    st.caption(f"Current sum: {current.sum():.3f}  ·  Target sum: {w_opt.sum():.3f}")
    if abs(current.sum() - 1.0) > 0.02:
        st.warning("Gewichtssumme ist nicht ~1.00 – bitte vor Trade planen normalisieren.")

    delta = (w_opt - current).rename("Delta (target-current)")
    last_prices = px_base.iloc[-1].reindex(w_opt.index).fillna(0.0)
    notional = (delta * float(portfolio_value)).rename(f"Trade Notional ({base_ccy})")

    fr_names = {ASSET_MAP[k]:k for k in ASSET_MAP}
    plan = pd.concat([
        current.rename("Current"),
        w_opt.rename("Target"),
        delta, last_prices.rename("Last Price"), notional
    ], axis=1).round(6)
    plan.index = [fr_names.get(idx, idx) for idx in plan.index]
    st.dataframe(plan, use_container_width=True)

    buf = io.BytesIO(); plan.to_csv(buf, index=True)
    st.download_button("Download Rebalance Plan (CSV)", data=buf.getvalue(),
                       file_name="rebalance_plan.csv", mime="text/csv")

# ---------- Paper Broker (stub) ----------
@dataclass
class BrokerState:
    cash: float = 100000.0
    positions: Dict[str, float] = field(default_factory=dict)  # shares
    history: List[Dict] = field(default_factory=list)

def broker_mark_to_market(state: BrokerState, prices: pd.Series) -> Tuple[float, float]:
    pos_val = sum(state.positions.get(t,0.0) * prices.get(t, np.nan) for t in state.positions.keys())
    equity = state.cash + pos_val
    return pos_val, equity

if "broker" not in st.session_state:
    st.session_state.broker = BrokerState(cash=float(portfolio_value))

with tab_broker:
    st.subheader("Paper Broker — Orders, Positions & P&L")
    b: BrokerState = st.session_state.broker
    prices = px_base.iloc[-1]
    pos_val, equity = broker_mark_to_market(b, prices)
    c1,c2,c3 = st.columns(3)
    c1.metric("Cash", f"{b.cash:,.2f} {base_ccy}")
    c2.metric("Positions Value", f"{pos_val:,.2f} {base_ccy}")
    c3.metric("Equity", f"{equity:,.2f} {base_ccy}")

    col1, col2, col3, col4 = st.columns([1.2,1,1,1])
    with col1: sym = st.selectbox("Symbol", list(px_base.columns))
    with col2: side = st.selectbox("Side", ["BUY","SELL"])
    with col3: qty = st.number_input("Qty (shares)", min_value=0.0, value=0.0, step=1.0)
    with col4:
        if st.button("Submit Order"):
            px_now = prices.get(sym, np.nan)
            if np.isnan(px_now) or qty<=0: st.warning("Ungültiger Preis oder Menge.")
            else:
                cost = qty * px_now
                if side=="BUY":
                    if b.cash >= cost:
                        b.cash -= cost; b.positions[sym] = b.positions.get(sym,0.0) + qty
                        b.history.append({"ts":datetime.now().isoformat(),"sym":sym,"side":side,"qty":qty,"px":px_now})
                        st.success("Order ausgeführt.")
                    else: st.error("Nicht genug Cash.")
                else:
                    have = b.positions.get(sym,0.0)
                    if have >= qty:
                        b.positions[sym] = have - qty; b.cash += cost
                        b.history.append({"ts":datetime.now().isoformat(),"sym":sym,"side":side,"qty":qty,"px":px_now})
                        st.success("Order ausgeführt.")
                    else: st.error("Nicht genug Bestand.")

    st.markdown("**Open Positions**")
    if b.positions:
        df_pos = pd.DataFrame.from_dict(b.positions, orient="index", columns=["Shares"])
        df_pos["Last Price"] = [prices.get(k,np.nan) for k in df_pos.index]
        df_pos["Market Value"] = df_pos["Shares"] * df_pos["Last Price"]
        st.dataframe(df_pos.round(4), use_container_width=True)
    else:
        st.write("Keine Positionen.")

    st.markdown("**Order History**")
    if b.history: st.dataframe(pd.DataFrame(b.history), use_container_width=True)
    else: st.write("Keine Orders bisher.")

# ---------- AI Insights (light) ----------
with tab_ai:
    st.subheader("AI Insights (Explain & Suggest)")
    notes = []
    crypto_assets = [c for c in w_opt.index if c in CRYPTOS]
    crypto_exp = float(w_opt.loc[crypto_assets].clip(lower=0).sum()) if crypto_assets else 0.0
    top_w = w_opt.sort_values(ascending=False).head(3)
    notes.append("Top Weights: " + ", ".join([f"{c} {v:.0%}" for c,v in top_w.items()]) + ".")
    notes.append(f"Crypto Exposure: {crypto_exp:.0%} (Cap {crypto_cap:.0%}).")
    notes.append(f"Regime: **{regime}** · Risk model: {('EWMA' if use_ewma_now else 'Ledoit-Wolf')} · Vol*-target: {target_vol_eff:.0%}.")
    notes.append(f"Leverage live: {scale_live:.2f}×; Drawdown guard thr {dd_guard_thr:.0%}, strength {dd_guard_strength:.0%}.")
    notes.append(f"Risk snapshot → VaR1d: {VaR_eff:.2%}, ES1d: {ES_eff:.2%}, MaxDD: {MDD_eff:.2%}.")
    if hedge_overlay_on: notes.append(f"Hedge overlay active: GLD {hedge_add:.0%}, FX hedge {fx_hedge_ratio:.0%} (base {base_ccy}).")
    if crypto_exp > crypto_cap + 1e-6: notes.append("🔧 Reduce crypto weights to respect cap.")
    if MDD_eff < -0.25: notes.append("🛡 High DD: lower target vol, increase hedge, or add bonds/gold.")
    if sr_eff < 0.6 and ann_vol_eff > target_vol_eff * 0.9: notes.append("⚖️ Low Sharpe with high vol → try ERC/Min-CVaR, tweak momentum.")
    try:
        from sklearn.cluster import KMeans
        feat = pd.DataFrame({
            "vol": rets_base.rolling(20).std().iloc[-90:].mean(),
            "mom": (px_base.iloc[-1]/px_base.iloc[-63]-1.0).reindex(rets_base.columns)
        }).fillna(0.0)
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(feat.values)
        cl = km.labels_; high_idx = np.argmax([feat.values[cl==i,0].mean() for i in [0,1]])
        high_cluster_members = feat.index[cl==high_idx].tolist()
        notes.append("🤖 ML Regime (cluster-high vol): " + ", ".join(high_cluster_members[:6]) + ("…" if len(high_cluster_members)>6 else ""))
    except Exception:
        notes.append("🤖 ML Regime: (optional) install scikit-learn for clustering insights.")
    st.write("\n\n".join(f"- {m}" for m in notes))

# ---------- Report ----------
with tab_report:
    st.subheader("One-Click HTML Report")
    fr_names = {ASSET_MAP[k]:k for k in ASSET_MAP}
    weights_named = {fr_names.get(k,k): float(v) for k,v in w_opt.round(4).to_dict().items()}
    proxy_note = []
    if "QQQ" in tickers: proxy_note.append("NASDAQ → QQQ")
    if "URTH" in tickers: proxy_note.append("MSCI World → URTH")
    if "UFO"  in tickers: proxy_note.append("STARLINK/SpaceX → UFO (space economy ETF)")
    html = f"""
    <html><head><meta charset="utf-8"><title>ALADDIN 3.2 Report</title></head>
    <body style="font-family:Inter,system-ui,sans-serif;background:#0c0f14;color:#e5e7eb;">
      <h2>ALADDIN 3.2 — Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})</h2>
      <p><b>Universe:</b> {', '.join(friendly_selection)}</p>
      <p><b>Proxies:</b> {'; '.join(proxy_note) if proxy_note else '—'}</p>
      <p><b>Model:</b> {opt_label} | <b>Regime:</b> {regime} | <b>Caps:</b> SA {single_cap:.0%} / Eq {equity_cap:.0%} / Crypto {crypto_cap:.0%}
         | <b>RB:</b> Crypto {rb_crypto:.0%}, Equity {rb_equity:.0%}
         | <b>Leverage:</b> {scale_live:.2f}× | <b>Base:</b> {base_ccy} | <b>FXH:</b> {fx_hedge_ratio:.0%}</p>
      <h3>Weights</h3>
      <pre>{json.dumps(weights_named, indent=2)}</pre>
      <h3>Key Metrics (post-scale)</h3>
      <ul>
        <li>Annualized Return: {ann_ret_eff:.2%}</li>
        <li>Annualized Volatility: {ann_vol_eff:.2%}</li>
        <li>Sharpe: {sr_eff:.2f}</li>
        <li>Skew: {skew:.2f}, Kurtosis (excess): {kurt:.2f}</li>
        <li>VaR({int(alpha*100)}%): {VaR_eff:.2%}</li>
        <li>CVaR({int(alpha*100)}%) ~ {ES_eff:.2%}</li>
        <li>Max Drawdown: {MDD_eff:.2%}</li>
      </ul>
      <p style="color:#94a3b8;">Hinweis: BL/HRP/Min-CVaR/ERC benötigen optionale Libraries; bei Fehlen wird ein robustes Fallback genutzt.</p>
    </body></html>
    """
    st.download_button("Download HTML Report", data=html.encode("utf-8"),
                       file_name="aladdin_report.html", mime="text/html")

# ================== END ==================

