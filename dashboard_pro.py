# ==========================================================
# ALADDIN 3.1 — MAXI HEDGEFUND (Single-file Streamlit App)
# Minimal Neon UI • Multi-Optimizer • Black-Litterman (cost-aware)
# Walk-Forward (TC/Liquidity) • EWMA/Regime • Target-Vol/Lev
# Momentum Tilt • Purged K-Fold CV • Deflated Sharpe
# Stress (2008/2020/2022) • Rebalance Planner • Paper Broker
# Presets • Report • AI-Insights (light)
# UI: Only friendly market names (no raw tickers like SPY/QQQ shown)
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
import requests  # Telegram

warnings.filterwarnings("ignore")

# ---------- Page / Theme ----------
st.set_page_config(page_title="ALADDIN 3.1 — MAXI HEDGEFUND", layout="wide")

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
PLOTLY_CFG = {"displaylogo": False,
              "toImageButtonOptions": {"format":"png","filename":"aladdin_chart",
                                       "height":450,"width":900,"scale":2}}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
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
.block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
h1,h2,h3 { letter-spacing:.3px; }
.kpi-card { background: var(--card); border:1px solid var(--border); border-radius:14px;
  padding:16px 18px; box-shadow:0 12px 30px rgba(0,0,0,.30);
  transition: box-shadow .25s ease, border-color .25s ease; }
.kpi-card:hover { border-color: rgba(34,211,238,.45);
  box-shadow: 0 0 0 1px rgba(34,211,238,.25), 0 20px 40px rgba(0,0,0,.35); }
.kpi-big { font-size:2.0rem; font-weight:800; color:#eaf7ff; line-height:1.1; }
.kpi-label { color:#95a3b3; font-size:.92rem; }
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent); margin:12px 0 18px; }
.smallnote { color:#93a1b1; font-size:.9rem; }
.stDataFrame { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.stButton>button { border-radius:10px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; position: sticky; top: 0; z-index: 10; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.06); border-radius: 10px; padding: 8px 12px; }
.stTabs [aria-selected="true"] { background: rgba(34,211,238,.14); border:1px solid rgba(34,211,238,.35); }
.section { border:1px solid var(--border); border-radius:14px; padding:16px; background: var(--card); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="background-grid"></div>', unsafe_allow_html=True)

# ---- Telegram Alerts (simple) ----
def send_telegram(msg: str) -> bool:
    """Sendet msg an deinen Telegram-Chat. Erwartet TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID in st.secrets."""
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]

    except Exception:
        st.error("Telegram Secrets fehlen. In Streamlit Cloud → Settings → Secrets setzen.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Telegram Fehler: {e}")
        return False

# ---------- Assets (friendly -> ticker map) ----------
ASSET_MAP = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Gold": "GLD",
    "NVIDIA": "NVDA",
    "NASDAQ": "QQQ",         # Nasdaq-100 proxy
    "S&P 500": "SPY",        # S&P 500 proxy
    "MSCI World ETF": "URTH",
    "STARLINK (SPACE X)": "UFO"
}
FRIENDLY_ORDER = list(ASSET_MAP.keys())

# Simple group tags for caps
CRYPTO_KEYS = ("BTC-USD", "ETH-USD", "SOL-USD")
EQUITY_TICKERS = {"AAPL", "TSLA", "NVDA", "QQQ", "SPY", "URTH", "UFO"}
CRYPTO_TICKERS = CRYPTO_KEYS

# Helpers for label mapping
TICKER_TO_FRIENDLY = {}
for k, v in ASSET_MAP.items():
    TICKER_TO_FRIENDLY.setdefault(v, k)

def labelize(idx: List[str]) -> List[str]:
    return [TICKER_TO_FRIENDLY.get(x, x) for x in idx]

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
    return float(series.mean())*freq, float(series.std(ddof=0))*math.sqrt(freq)

def sharpe_ratio(series: pd.Series, rf: float = 0.0, freq: int = 252) -> float:
    if series.empty: return 0.0
    ex = series - (rf/freq); vol = float(ex.std(ddof=0))
    return 0.0 if vol==0 else float(ex.mean()/vol*math.sqrt(freq))

def hist_var_es(series: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    s = np.sort(series.dropna().values)
    if s.size == 0: return 0.0, 0.0
    k = max(0, int((1-alpha)*len(s)) - 1)
    var = s[k]; es = s[:k+1].mean() if k>=0 else var
    return float(var), float(es)

def higher_moments(series: pd.Series) -> Tuple[float,float]:
    x = series.dropna()
    if len(x)==0 or x.std(ddof=0)==0: return 0.0, 0.0
    skew = float(((x-x.mean())**3).mean()/(x.std(ddof=0)**3))
    kurt = float(((x-x.mean())**4).mean()/(x.std(ddof=0)**4)) - 3.0
    return skew, kurt

def max_drawdown(cum: pd.Series) -> float:
    if cum.empty: return 0.0
    return float(((cum/cum.cummax())-1.0).min())

def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    X = returns.fillna(0.0).values
    n, k = X.shape
    seed = min(252, max(60, n//4))
    S = np.cov(X[:seed].T) if n>=seed else np.eye(k)
    for t in range(seed, n):
        x = X[t].reshape(-1,1)
        S = lam*S + (1-lam)*(x@x.T)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def regime_tag(returns: pd.DataFrame, lookback: int = 21, thr: float = 0.02) -> str:
    if returns.empty: return "unknown"
    vol = returns.rolling(lookback).std().dropna().iloc[-1].mean()
    return "high-vol" if vol>thr else "normal"

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
            df = yf.download(tickers, start=start, period=period,
                             auto_adjust=True, progress=False,
                             group_by="ticker", threads=True)
            if isinstance(df, pd.DataFrame) and len(df)>0:
                return df
        except Exception as e:
            last_err = e
        time.sleep(sleep*(k+1))
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
    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    px = px.ffill().bfill()
    pct = px.pct_change(); bad = pct.abs() > clamp_thr
    if bad.any().any():
        px = px.copy()
        for col in px.columns:
            idx = bad[col][bad[col]].index
            for t in idx:
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
        px  = df.xs("Adj Close", axis=1, level=1) if "Adj Close" in df.columns.get_level_values(1) else df.xs("Close", axis=1, level=1)
        vol = df.xs("Volume",    axis=1, level=1)
    else:
        px  = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        vol = df["Volume"] if "Volume" in df.columns else pd.DataFrame(index=px.index)
    px  = px.reindex(columns=[t for t in tickers if t in px.columns]).ffill().bfill()
    vol = vol.reindex(columns=[t for t in tickers if t in vol.columns]).fillna(0)
    pct = px.pct_change(); bad = pct.abs() > clamp_thr
    if bad.any().any():
        px = px.copy()
        for col in px.columns:
            idx = bad[col][bad[col]].index
            for t in idx:
                prev = px[col].shift(1).get(t, np.nan)
                nxt  = px[col].shift(-1).get(t, np.nan)
                rep = np.nanmean([prev, nxt])
                if not np.isnan(rep): px.at[t, col] = rep
    return px.dropna(how="all"), vol.fillna(0)

# ---------- Optimizers ----------
def _project_to_simplex_with_bounds(w: np.ndarray, lb: float, ub: float) -> np.ndarray:
    w = np.clip(w, lb, ub); s = w.sum()
    if s <= 0: w[:] = 1.0/len(w)
    else:
        w = w/s; w = np.clip(w, lb, ub); w = w/w.sum()
    return w

def inverse_variance_weights(px_hist: pd.DataFrame, lb: float, ub: float) -> pd.Series:
    n = px_hist.shape[1]
    if n==0: return pd.Series(dtype=float)
    if px_hist.shape[0] < 40: return pd.Series(np.ones(n)/n, index=px_hist.columns, dtype=float)
    rets = to_returns(px_hist)
    var = rets.var().replace([np.inf,-np.inf], np.nan).fillna(1.0)
    iv = 1.0/(var + 1e-12)
    w = _project_to_simplex_with_bounds(iv.values, lb, ub)
    return pd.Series(w, index=px_hist.columns, dtype=float)

def optimizer_min_vol(px_hist: pd.DataFrame, lb: float, ub: float, use_ewma: bool) -> Tuple[pd.Series, str]:
    try:
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.efficient_frontier import EfficientFrontier
        S = ewma_cov(to_returns(px_hist), lam=0.94) if use_ewma else __import__("pypfopt.risk_models", fromlist=['CovarianceShrinkage']).risk_models.CovarianceShrinkage(px_hist).ledoit_wolf()
        mu = mean_historical_return(px_hist)
        ef = EfficientFrontier(mu, S, weight_bounds=(lb, ub))
        w = pd.Series(ef.min_volatility())
        w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0)
        if abs(float(w.sum())-1.0) > 1e-9: w = w/w.sum()
        return w, ("Min-Vol (EWMA)" if use_ewma else "Min-Vol (Ledoit-Wolf)")
    except Exception:
        return inverse_variance_weights(px_hist, lb, ub), "Inverse-Variance"

def optimizer_hrp(px_hist: pd.DataFrame, lb: float, ub: float) -> Tuple[pd.Series, str]:
    try:
        from pypfopt.hierarchical_risk_parity import HRPOpt
        rets = to_returns(px_hist); hrp = HRPOpt(rets)
        w = pd.Series(hrp.optimize()); w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0)
        if abs(float(w.sum())-1.0) > 1e-9: w = w/w.sum()
        w = pd.Series(_project_to_simplex_with_bounds(w.values, lb, ub), index=w.index)
        return w, "HRP"
    except Exception:
        return inverse_variance_weights(px_hist, lb, ub), "Inverse-Variance"

def optimizer_cvar(returns: pd.DataFrame, alpha: float, lb: float, ub: float, allow_short: bool) -> Tuple[pd.Series, str]:
    try:
        import cvxpy as cp
        R = returns.values; T, N = R.shape
        w = cp.Variable(N); z = cp.Variable(T); v = cp.Variable(1)
        tau = 1 - alpha
        obj = v + (1.0/(tau*T))*cp.sum(z)
        cons = []
        if allow_short:
            cons += [cp.sum(cp.abs(w)) == 1.0, w >= -ub, w <= ub]
        else:
            cons += [cp.sum(w) == 1.0, w >= lb, w <= ub]
        cons += [z >= 0, z >= -(R @ w) - v]
        cp.Problem(cp.Minimize(obj), cons).solve(solver=cp.ECOS, verbose=False, warm_start=True)
        w_opt = pd.Series(np.array(w.value).ravel(), index=returns.columns).fillna(0.0)
        w_opt = pd.Series(_project_to_simplex_with_bounds(w_opt.values, lb, ub), index=w_opt.index)
        return w_opt, "Min-CVaR"
    except Exception:
        return inverse_variance_weights(returns.cumsum(), lb, ub), "Inverse-Variance"

# ---------- Black-Litterman (cost-aware) ----------
def black_litterman(px_hist: pd.DataFrame, base_cov: pd.DataFrame, mkt_w: pd.Series,
                    tau: float, P: np.ndarray, Q: np.ndarray, omega: Optional[np.ndarray]=None) -> Tuple[pd.Series, pd.DataFrame]:
    S = base_cov.values; w = mkt_w.values.reshape(-1,1)
    Pi = (S @ w).ravel()
    if omega is None: omega = np.diag(np.diag(P @ (tau*S) @ P.T))
    A = np.linalg.inv(tau*S); M = A + P.T @ np.linalg.inv(omega) @ P
    b = A @ Pi + P.T @ np.linalg.inv(omega) @ Q
    mu_bl = pd.Series(np.linalg.solve(M, b), index=px_hist.columns)
    cov_bl = pd.DataFrame(np.linalg.inv(M), index=px_hist.columns, columns=px_hist.columns)
    return mu_bl, cov_bl

def optimizer_black_litterman(px_hist: pd.DataFrame, lb: float, ub: float,
                              views_df: pd.DataFrame, tau: float=0.025, cost_bps: float=5.0) -> Tuple[pd.Series, str]:
    rets = to_returns(px_hist); S = ewma_cov(rets); w_mkt = inverse_variance_weights(px_hist, 0, 1.0)
    cols = list(px_hist.columns); P_list, Q_list = [], []
    for _, r in views_df.iterrows():
        asset = r.get("Asset",""); vtype = r.get("Type","abs"); val = float(r.get("Value",0.0))
        if asset not in cols: continue
        row = np.zeros(len(cols)); j = cols.index(asset)
        if vtype == "abs":
            row[j] = 1.0; P_list.append(row); Q_list.append(val)
        elif vtype == "relMkt":
            row[j] = 1.0; others = [i for i in range(len(cols)) if i!=j]
            row[others] = -w_mkt.values[others]/max(1e-12, w_mkt.drop(asset).sum())
            P_list.append(row); Q_list.append(val)
        else:
            row[j] = 1.0; others = [i for i in range(len(cols)) if i!=j]
            row[others] = -1.0/len(others); P_list.append(row); Q_list.append(val)
    if not P_list:
        return optimizer_min_vol(px_hist, lb, ub, use_ewma=True)
    P = np.vstack(P_list); Q = np.array(Q_list)
    mu_bl, cov_bl = black_litterman(px_hist, S, w_mkt, tau=tau, P=P, Q=Q)
    try:
        from pypfopt.efficient_frontier import EfficientFrontier
        ef = EfficientFrontier(mu_bl, cov_bl, weight_bounds=(lb, ub))
        w = pd.Series(ef.max_sharpe()); w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0); w = w/w.sum()
    except Exception:
        inv = 1.0/(np.diag(cov_bl)+1e-9)
        pref = inv*(mu_bl.values - mu_bl.values.mean())
        w = pd.Series(_project_to_simplex_with_bounds(pref, lb, ub), index=px_hist.columns)
    gamma = cost_bps/10000.0
    w = (1-gamma)*w + gamma*w_mkt.reindex_like(w).fillna(0.0)
    w = w/w.sum()
    return w, "Black-Litterman (cost-aware)"

# ---------- Caps & Constraints ----------
def apply_caps(weights: pd.Series, crypto_cap: float, equity_cap: float, single_cap: float) -> pd.Series:
    w = weights.copy().fillna(0.0).astype(float)
    if single_cap < 1.0:
        w = w.clip(upper=single_cap)
        if w.sum() > 0:
            w /= w.sum()
    crypto_cols = [c for c in w.index if c in CRYPTO_KEYS]
    if crypto_cols:
        cs = float(w.loc[crypto_cols].clip(lower=0).sum())
        if crypto_cap <= 0:
            w.loc[crypto_cols] = 0.0
            if w.sum() > 0:
                w /= w.sum()
        elif cs > crypto_cap and cs > 0:
            scale = crypto_cap / cs
            w.loc[crypto_cols] = w.loc[crypto_cols] * scale
            if w.sum() > 0:
                w /= w.sum()
    eq_cols = [c for c in w.index if c in EQUITY_TICKERS]
    if eq_cols and equity_cap < 1.0:
        es = float(w.loc[eq_cols].clip(lower=0).sum())
        if es > equity_cap and es > 0:
            scale = equity_cap / es
            w.loc[eq_cols] = w.loc[eq_cols] * scale
            if w.sum() > 0:
                w /= w.sum()
    return w

# ---------- UI Header ----------
colH1, colH2 = st.columns([0.8, 0.2])
with colH1:
    st.title("ALADDIN 3.1 — MAXI HEDGEFUND")
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

# ---- Alerts (Telegram) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Alerts")
if st.sidebar.button("🔔 Telegram Test-Alert senden", use_container_width=True):
    ok = send_telegram("ALADDIN: Test-Alert ✅")
    if ok:
        st.sidebar.success("Test-Alert gesendet.")
    else:
        st.sidebar.error("Konnte Test-Alert nicht senden (Secrets prüfen).")

# ---- Universe ----
DEFAULT_FRIENDLY = [
    "BTCUSD","ETHUSD","SOLUSD",
    "Apple","Tesla","Gold","NVIDIA",
    "NASDAQ","S&P 500","MSCI World ETF","STARLINK (SPACE X)"
]
friendly_selection = st.sidebar.multiselect(
    "Universe (choose from your set)",
    options=FRIENDLY_ORDER,
    default=DEFAULT_FRIENDLY
)
friendly_selection = [s.strip() for s in friendly_selection]
friendly_selection = list(dict.fromkeys(friendly_selection))
unknown = [s for s in friendly_selection if s not in ASSET_MAP]
if unknown:
    st.sidebar.warning(f"Ignored unknown items in universe: {unknown}")
    friendly_selection = [s for s in friendly_selection if s in ASSET_MAP]
tickers = [ASSET_MAP[s] for s in friendly_selection]

years = st.sidebar.slider("Years of history", 2, 15, value=5)
outlier_thr = st.sidebar.slider("Outlier clamp (1d move)", 0.20, 0.80, 0.40, 0.05)
min_w = st.sidebar.slider("Min weight per asset", 0.00, 0.20, 0.00, step=0.01)
max_w = st.sidebar.slider("Max weight per asset", 0.10, 0.70, 0.35, step=0.01)
single_cap = st.sidebar.slider("Single-asset cap", 0.10, 1.00, 0.45, step=0.05)
equity_cap = st.sidebar.slider("Equity cap (group)", 0.10, 1.00, 0.75, step=0.05)
crypto_cap = st.sidebar.slider("Crypto cap (group)", 0.00, 0.80, 0.20, step=0.01)
alpha = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.90, 0.99, 0.95, step=0.01)
allow_short = st.sidebar.checkbox("Allow short (experimental)", value=False)
portfolio_value = st.sidebar.number_input("Portfolio value (EUR)", 1000.0, 1e9, 100000.0, step=1000.0, format="%.2f")
target_vol = st.sidebar.slider("Target annualized vol", 0.05, 0.40, 0.12, step=0.01)
max_leverage = st.sidebar.slider("Max gross leverage", 1.0, 3.0, 1.5, step=0.1)
mom_strength = st.sidebar.slider("Momentum tilt strength", 0.0, 1.0, 0.30, step=0.05)
mom_lb = st.sidebar.selectbox("Momentum lookback", ["6M","9M","12M"], index=2)
opt_choice = st.sidebar.selectbox("Optimizer", ["Min-Vol", "HRP", "Min-CVaR", "Black-Litterman"], index=0)
use_ewma = st.sidebar.checkbox("Use EWMA cov in high-vol regime (auto)", value=True)
wf_reb = st.sidebar.selectbox("Backtest rebalance", ["Monthly","Weekly"], index=0)
wf_cost_bps = st.sidebar.number_input("Base TC (bps per turnover)", 0, 300, 10, step=5)
wf_impact_k = st.sidebar.slider("Impact coef (sqrt model)", 0.0, 50.0, 8.0, step=0.5)
wf_turnover_cap = st.sidebar.slider("Turnover cap per rebalance", 0.05, 1.0, 0.35, step=0.05)
wf_min_trade = st.sidebar.slider("Min trade per asset (Δw)", 0.000, 0.050, 0.005, step=0.005)
wf_adv_days = st.sidebar.slider("ADV window (days)", 10, 60, 20, step=5)
wf_max_pct_adv = st.sidebar.slider("Max notional per rebalance as %ADV", 1.0, 50.0, 10.0, step=1.0)
mc_block_bootstrap = st.sidebar.checkbox("MC: Block-bootstrap instead of Gaussian", value=True)
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

# Presets
st.sidebar.markdown("**Presets**")
if st.sidebar.button("Save current preset"):
    preset = {
        "friendly_selection": friendly_selection, "years": years,
        "min_w": min_w, "max_w": max_w, "single_cap": single_cap,
        "equity_cap": equity_cap, "crypto_cap": crypto_cap,
        "alpha": alpha, "allow_short": allow_short,
        "portfolio_value": portfolio_value, "opt_choice": opt_choice,
        "use_ewma": use_ewma, "wf_reb": wf_reb, "wf_cost_bps": wf_cost_bps,
        "wf_turnover_cap": wf_turnover_cap, "wf_min_trade": wf_min_trade,
        "target_vol": target_vol, "max_leverage": max_leverage,
        "mom_strength": mom_strength, "mom_lb": mom_lb, "wf_adv_days": wf_adv_days,
        "wf_max_pct_adv": wf_max_pct_adv, "wf_impact_k": wf_impact_k,
        "outlier_thr": outlier_thr, "mc_block_bootstrap": mc_block_bootstrap,
        "mc_block": mc_block, "mc_horizon": mc_horizon, "mc_sims": mc_sims
    }
    st.sidebar.download_button("Download preset.json", data=json.dumps(preset, indent=2).encode(),
                               file_name="preset.json", mime="application/json")
uploaded = st.sidebar.file_uploader("Load preset.json", type=["json"], key="preset_upl")
if uploaded and "preset_loaded_once" not in st.session_state:
    try:
        preset = json.load(uploaded)
        st.session_state.update(preset)
        st.sidebar.success("Preset loaded. Bitte Seite ggf. neu laden (R).")
        st.session_state["preset_loaded_once"] = True
    except Exception as e:
        st.sidebar.error(f"Preset error: {e}")

# ---------- Data ----------
if len(tickers) == 0:
    st.warning("Bitte mindestens einen Markt wählen."); st.stop()

try:
    px, vol_df = load_ohlcv(tickers, years=years, clamp_thr=outlier_thr)
    if px.empty:
        st.error("Keine Preisdaten geladen — prüfe Ticker."); st.stop()
    rets = to_returns(px).dropna(how="all")
except Exception as e:
    st.error(f"Datenfehler: {e}"); st.stop()

# Data Quality checks
dq_missing = float(px.isna().mean().mean())
dq_min_hist = int(px.notna().sum().min())
dq_last = px.index.max()
dq_last_str = dq_last.date().isoformat() if hasattr(dq_last,"date") else str(dq_last)
if dq_missing > 0.02 or dq_min_hist < 250:
    st.warning(f"Data Quality: missing={dq_missing:.1%}, min_history={dq_min_hist} Tage. Latest={dq_last_str}")

# ---------- Optimize (target) ----------
regime = regime_tag(rets)
use_ewma_now = (use_ewma and regime=="high-vol")

default_views = pd.DataFrame({"Asset":[px.columns[0] if len(px.columns) else ""],
                              "Type":["abs"], "Value":[0.08]})
views_df = None
if opt_choice == "Black-Litterman":
    st.sidebar.markdown("**Black-Litterman Views**")
    st.sidebar.caption("Type: abs = expected annual return; rel = vs equal basket; relMkt = vs market weights")
    choices_df = default_views.copy()
    choices_df["Asset"] = choices_df["Asset"].map(lambda t: t)
    views_df = st.sidebar.data_editor(choices_df, num_rows="dynamic", use_container_width=True)

imported = st.session_state.get("import_weights")
if imported is not None:
    imp = imported.reindex(px.columns).fillna(0.0).clip(lower=0.0)
    if imp.sum()>0: imp /= imp.sum()
    imp = apply_caps(imp, crypto_cap, equity_cap, single_cap)
else:
    imp = None

if opt_choice == "Min-Vol":
    w_opt, opt_label = optimizer_min_vol(px, lb=min_w, ub=max_w, use_ewma=use_ewma_now)
elif opt_choice == "HRP":
    w_opt, opt_label = optimizer_hrp(px, lb=min_w, ub=max_w)
elif opt_choice == "Min-CVaR":
    w_opt, opt_label = optimizer_cvar(rets, alpha=alpha, lb=min_w, ub=max_w, allow_short=allow_short)
else:
    try:
        w_opt, opt_label = optimizer_black_litterman(px, lb=min_w, ub=max_w,
            views_df=views_df if views_df is not None else pd.DataFrame(columns=["Asset","Type","Value"]))
    except Exception:
        w_opt, opt_label = optimizer_min_vol(px, lb=min_w, ub=max_w, use_ewma=True)

w_opt = w_opt.reindex(px.columns).fillna(0.0)
if imp is not None and imp.sum()>0:
    w_opt = 0.7*w_opt + 0.3*imp

w_opt = apply_caps(w_opt, crypto_cap, equity_cap, single_cap)
if w_opt.sum()>0: w_opt = w_opt/w_opt.sum()

lb_map = {"6M":126, "9M":189, "12M":252}; lb_days = lb_map[mom_lb]
mom = (px/px.shift(lb_days) - 1.0).iloc[-1].reindex(w_opt.index).fillna(0.0)
mom_score = (mom.rank(pct=True) - 0.5)
w_tilt = w_opt * (1 + mom_strength*mom_score)
if w_tilt.sum()>0: w_opt = (w_tilt/w_tilt.sum()).fillna(0.0)

# ---------- Portfolio series & risk (post-scale) ----------
port = (rets.fillna(0.0) @ w_opt.values).rename("Portfolio")
cum = (1.0 + port).cumprod()
ann_ret, ann_vol = annualize_stats(port)
sr = sharpe_ratio(port)
VaR, ES = hist_var_es(port, alpha=alpha)
MDD = max_drawdown(cum)
skew, kurt = higher_moments(port)

eps = 1e-8
scale_live = min(max_leverage, (target_vol/max(eps, ann_vol)))
port_eff = port * scale_live
cum_eff  = (1.0 + port_eff).cumprod()
ann_ret_eff, ann_vol_eff = annualize_stats(port_eff)
sr_eff  = sharpe_ratio(port_eff)
VaR_eff, ES_eff = hist_var_es(port_eff, alpha=alpha)
MDD_eff = max_drawdown(cum_eff)
roll_sharpe = port_eff.rolling(60).apply(lambda x: 0.0 if np.std(x)==0 else (np.mean(x)/np.std(x))*np.sqrt(252), raw=True)

# ---------- Tabs ----------
tab_overview, tab_risk, tab_backtest, tab_cv, tab_factors, tab_stress, tab_trades, tab_broker, tab_ai, tab_report = st.tabs(
    ["Overview", "Risk", "Backtest", "CV & DS", "Factors", "Stress", "Trades", "Paper Broker", "AI Insights", "Report"]
)

# ---------- Overview ----------
def kpi(label, value):
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-big">{value}</div></div>""", unsafe_allow_html=True)

with tab_overview:
    a,b,c,d = st.columns(4)
    with a: kpi("Portfolio Index", f"{cum_eff.iloc[-1]:.2f}x")
    with b: kpi("Sharpe (post-scale)", f"{sr_eff:.2f}")
    with c: kpi(f"VaR({int(alpha*100)}%) 1d", f"{VaR_eff:.2%}")
    with d: kpi("Max Drawdown", f"{MDD_eff:.2%}")
    st.caption(f"Optimizer: {opt_label} · Caps: Single {single_cap:.0%} / Equities {equity_cap:.0%} / Crypto {crypto_cap:.0%} · Regime: {regime} · Leverage: {scale_live:.2f}×")

    sa, sb, sc, sd = st.columns(4)
    with sa: st.plotly_chart(sparkline((1+port_eff).cumprod()), use_container_width=True, config={"displayModeBar": False})
    with sb: st.plotly_chart(sparkline(roll_sharpe.fillna(0)), use_container_width=True, config={"displayModeBar": False})
    with sc: st.plotly_chart(sparkline((cum_eff/cum_eff.cummax()-1.0).fillna(0)), use_container_width=True, config={"displayModeBar": False})
    with sd: st.plotly_chart(sparkline(rets.mean(axis=1).rolling(20).mean().fillna(0)), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    L, R = st.columns([1.25, 1.0])
    with L:
        st.subheader("Portfolio Overview (target)")
        ytd = {t: rets[t].loc[rets.index.year==rets.index[-1].year].sum() if t in rets.columns else 0.0 for t in px.columns}
        w_opt_local = w_opt.astype(float)
        ytd_series = pd.to_numeric(pd.Series(ytd), errors="coerce").fillna(0.0)
        df_over = pd.DataFrame({"Weight": w_opt_local.round(4), "YTD Return": ytd_series.round(4)})
        df_over.index = labelize(df_over.index.tolist())
        st.dataframe(df_over, use_container_width=True)
        st.download_button("Export Weights (JSON)",
            data=json.dumps({k: float(v) for k,v in w_opt_local.round(6).to_dict().items()}, indent=2).encode(),
            file_name="weights.json", mime="application/json")
        if st.button("📲 Weights an Telegram senden"):
            nice = {TICKER_TO_FRIENDLY.get(k,k): f"{float(v):.1%}" for k,v in w_opt.items()}
            ok_msg = send_telegram(f"ALADDIN Weights {datetime.now():%Y-%m-%d %H:%M}:\n" + json.dumps(nice, indent=2))
            st.success("Gesendet.") if ok_msg else st.error("Senden fehlgeschlagen.")
    with R:
        st.subheader("Risk Metrics (post-scale)")
        df_risk = pd.DataFrame({
            "Annualized Return": [ann_ret_eff],
            "Annualized Volatility": [ann_vol_eff],
            "Sharpe": [sr_eff], "Skew": [skew], "Kurtosis (excess)": [kurt],
            f"VaR({int(alpha*100)}%)": [VaR_eff], f"CVaR({int(alpha*100)}%) ~": [ES_eff],
            "Max Drawdown": [MDD_eff], "Leverage (×)": [scale_live],
        }).round(4)
        st.dataframe(df_risk, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = rets.corr()
    im = ax.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(labelize(list(corr.columns)), rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(labelize(list(corr.index)))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    st.pyplot(fig)

# ---------- Risk ----------
def _gaussian_mc(rets: pd.DataFrame, w: pd.Series, scale: float, sims: int, horizon_d: int):
    mu = rets.mean().values; S = rets.cov().values
    R = np.random.multivariate_normal(mu, S, size=(sims, horizon_d))
    return (R @ w.values).sum(axis=1) * scale

def _block_bootstrap_mc(rets: pd.DataFrame, w: pd.Series, scale: float, sims: int, horizon_d: int, block_len: int):
    r = rets.values; T, N = r.shape; blocks = []
    starts = np.random.randint(0, max(1, T-block_len), size=(sims, math.ceil(horizon_d/block_len)))
    for s in range(sims):
        seq = []
        for idx in starts[s]:
            b = r[idx:idx+block_len]
            if len(b) < block_len: b = np.vstack([b, r[:(block_len-len(b))]])
            seq.append(b)
        path = np.vstack(seq)[:horizon_d]; blocks.append(path)
    R = np.stack(blocks, axis=0)
    return (R @ w.values[:,None]).sum(axis=1).ravel() * scale

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
    horizon_map = {"1d":1, "5d":5, "10d":10}; H = horizon_map[mc_horizon]
    try:
        if mc_block_bootstrap:
            port_sims = _block_bootstrap_mc(rets, w_opt, scale_live, mc_sims, H, mc_block)
        else:
            port_sims = _gaussian_mc(rets, w_opt, scale_live, mc_sims, H)
        VaR_mc = np.percentile(port_sims, (1-alpha)*100)
        ES_mc  = port_sims[port_sims<=VaR_mc].mean() if np.any(port_sims<=VaR_mc) else VaR_mc
        st.write(f"MC VaR({int(alpha*100)}%) {mc_horizon}: **{VaR_mc:.2%}**,  ES: **{ES_mc:.2%}**  ·  Sims: {mc_sims:,}  ({'Bootstrap' if mc_block_bootstrap else 'Gaussian'})")
    except Exception as e:
        st.warning(f"Monte Carlo nicht verfügbar: {e}")

# ---------- Backtest (Walk-Forward, TC/Liquidity) ----------
def monthly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    g = pd.Series(index=index, data=True)
    return g.groupby([index.year, index.month]).head(1).index

def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    g = pd.Series(index=index, data=True)
    return g.groupby([index.year, index.isocalendar().week]).head(1).index

def rebalance_backtest(px: pd.DataFrame, rets: pd.DataFrame, vol_df: pd.DataFrame, dates: pd.DatetimeIndex,
                       lb: float, ub: float, single_cap: float, equity_cap: float, crypto_cap: float,
                       base_bps: float, impact_k: float, adv_days: int, max_pct_adv: float,
                       turnover_cap: float, min_trade: float,
                       opt_choice: str, use_ewma: bool,
                       alpha: float, allow_short: bool, views_df: Optional[pd.DataFrame],
                       target_vol: float, max_leverage: float, mom_strength: float, mom_lb_days: int,
                       min_hist: int=63) -> Tuple[pd.Series, pd.Series]:
    w_prev = None; eq, tc_series = [], []; equity = 1.0
    adv = (vol_df.rolling(adv_days).mean() * px).fillna(0.0)  # ~ notional ADV
    for dt in rets.index:
        if dt in dates:
            px_hist = px.loc[:dt].dropna()
            if px_hist.shape[0] < min_hist:
                w = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)
            else:
                if opt_choice == "Min-Vol":   w = optimizer_min_vol(px_hist, lb, ub, use_ewma)[0]
                elif opt_choice == "HRP":     w = optimizer_hrp(px_hist, lb, ub)[0]
                elif opt_choice == "Min-CVaR":w = optimizer_cvar(to_returns(px_hist), alpha, lb, ub, allow_short)[0]
                else:
                    try: w = optimizer_black_litterman(px_hist, lb, ub, views_df=views_df if views_df is not None else pd.DataFrame(columns=["Asset","Type","Value"]))[0]
                    except Exception: w = optimizer_min_vol(px_hist, lb, ub, True)[0]
                w = apply_caps(w, crypto_cap, equity_cap, single_cap)
                if w.sum()>0: w = w/w.sum()
                mom = (px_hist/px_hist.shift(mom_lb_days)-1.0).iloc[-1].reindex(w.index).fillna(0.0)
                mom_score = (mom.rank(pct=True)-0.5); w_tilt = w*(1 + mom_strength*mom_score)
                if w_tilt.sum()>0: w = (w_tilt/w_tilt.sum()).fillna(0.0)

            if w_prev is not None:
                delta = (w - w_prev).abs()
                delta = delta.where(delta >= min_trade, 0.0)
                adv_now = adv.reindex(index=[dt]).ffill().iloc[0].reindex(w.index).replace(0, np.nan)
                desired_notional = delta * equity
                max_notional = (max_pct_adv/100.0)*adv_now
                scale = 1.0; mask = (~max_notional.isna()) & (max_notional>0)
                if mask.any():
                    ratio = (desired_notional[mask]/max_notional[mask]).max()
                    if ratio > 1.0: scale = 1.0/float(ratio)
                delta *= scale
                if float(delta.sum())>turnover_cap: delta = delta*(turnover_cap/float(delta.sum()))
                traded_notional = delta * equity
                imp = pd.Series(0.0, index=w.index, dtype=float)
                if mask.any():
                    z = traded_notional[mask]/max_notional[mask].replace(0,np.nan)
                    imp.loc[mask] = (base_bps/10000.0) + (impact_k/10000.0)*np.sqrt(z.clip(lower=0.0).fillna(0.0))
                else:
                    imp += (base_bps/10000.0)
                tc = float(imp.fillna(base_bps/10000.0).sum())
                equity *= (1.0 - tc); tc_series.append((dt, tc))
                w = w_prev + np.sign(w - w_prev)*delta; w = w/w.sum()
            w_prev = w.copy()

        if w_prev is None:
            w_prev = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)

        hist = rets.loc[:dt].tail(63).reindex(columns=w_prev.index).dot(w_prev.values)
        curr_vol = hist.std()*np.sqrt(252) if len(hist) else 0.0
        sf = min(max_leverage, (target_vol/max(1e-8, curr_vol))) if curr_vol>0 else 1.0
        r = float(rets.loc[dt].reindex(w_prev.index).fillna(0.0).dot(w_prev.values))
        equity *= (1.0 + sf*r); eq.append(equity)

    tc_series = pd.Series({d:v for d,v in tc_series}).reindex(rets.index).fillna(0.0).rename("TC")
    return pd.Series(eq, index=rets.index, name="Portfolio"), tc_series

with tab_backtest:
    st.subheader("Walk-Forward Backtest · Portfolio vs Benchmark (TC/Liquidity-aware)")
    bt_years = st.slider("Backtest years", 2, 15, value=min(5, years), key="bt_years")
    px_bt, vol_bt = load_ohlcv(tickers, years=bt_years, clamp_thr=outlier_thr)
    rets_bt = to_returns(px_bt).dropna()
    rebal_dates = monthly_dates(rets_bt.index) if wf_reb=="Monthly" else weekly_dates(rets_bt.index)

    eq, tc_series = rebalance_backtest(
        px_bt, rets_bt, vol_bt, rebal_dates,
        lb=min_w, ub=max_w, single_cap=single_cap, equity_cap=equity_cap, crypto_cap=crypto_cap,
        base_bps=wf_cost_bps, impact_k=wf_impact_k, adv_days=wf_adv_days, max_pct_adv=wf_max_pct_adv,
        turnover_cap=wf_turnover_cap, min_trade=wf_min_trade,
        opt_choice=opt_choice, use_ewma=use_ewma_now, alpha=alpha, allow_short=allow_short,
        views_df=views_df, target_vol=target_vol, max_leverage=max_leverage,
        mom_strength=mom_strength, mom_lb_days=lb_days
    )

    bench_choices_friendly = ["Gold", "NASDAQ", "S&P 500", "MSCI World ETF",
                              "BTCUSD", "ETHUSD", "SOLUSD", "Apple", "Tesla", "NVIDIA"]
    bench_label = st.selectbox("Benchmark", bench_choices_friendly, index=0, key="bench")
    bench_ticker = ASSET_MAP.get(bench_label, bench_label)

    try:
        bench_raw = _download_with_retry(bench_ticker, period=f"{bt_years}y")
        bench_px = _normalize_price_frame(bench_raw)
        if isinstance(bench_px, pd.DataFrame):
            if "Close" in bench_px.columns: 
                bench_px = bench_px["Close"]
            elif "Adj Close" in bench_px.columns:
                bench_px = bench_px["Adj Close"]
            else:
                bench_px = bench_px.iloc[:, 0]
        bench_px = bench_px.dropna()
    except Exception:
        bench_px = pd.Series(dtype=float)

    if bench_px.empty:
        st.warning(f"Keine Benchmark-Daten für '{bench_label}' (Ticker '{bench_ticker}'). Prüfe Mapping in ASSET_MAP.")
        bench_eq = pd.Series(index=eq.index, dtype=float)
    else:
        common = eq.index.intersection(bench_px.index)
        bench_px = bench_px.loc[common]
        bench_eq = bench_px / bench_px.iloc[0]

    port_eq  = eq.loc[bench_eq.index] / eq.loc[bench_eq.index].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_eq.index,  y=port_eq.values,  name="Portfolio", mode="lines"))
    if not bench_eq.empty:
        fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, name=bench_label, mode="lines"))
    fig.update_layout(title="Index (Start=1.0)", xaxis_title="Date", yaxis_title="Index", height=420, template="aladdin")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

# ---------- Purged K-Fold CV + Deflated Sharpe ----------
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
    emax = sr_mean + sr_std*(1 - 0.75*(np.log(np.log(n_trials)))/np.sqrt(2*np.log(n_trials)))
    return float((sr_hat - emax) * np.sqrt(max(1, T)))

with tab_cv:
    st.subheader("Purged K-Fold Cross-Validation & Deflated Sharpe")
    k = st.slider("Folds (K)", 3, 10, 5)
    embargo = st.slider("Embargo (days)", 0, 30, 10)
    if len(port_eff) > 200:
        srs = []
        for tr_idx, te_idx in time_series_folds(port_eff.index, k=k, embargo=embargo):
            te = port_eff.reindex(te_idx).dropna()
            srs.append(sharpe_ratio(te) if len(te)>0 else 0.0)
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

# ---------- Factors ----------
with tab_factors:
    st.subheader("Factor Attribution (Daily Returns Regression)")
    factor_map = {
        "S&P 500": ASSET_MAP["S&P 500"],   # SPY
        "NASDAQ": ASSET_MAP["NASDAQ"],     # QQQ
        "Gold": ASSET_MAP["Gold"],         # GLD
        "USD (proxy)": "UUP"
    }
    fac_px = load_prices(list(factor_map.values()), years=min(5, years), clamp_thr=outlier_thr)
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
    st.caption("1-Day shocks & historical scenarios mapped to portfolio (post-scale).")
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
    pick = st.multiselect("Assets to shock (optional override)",
                          options=list(px.columns), format_func=lambda t: TICKER_TO_FRIENDLY.get(t,t), default=[])
    shock_pct = st.slider("Shock size (manual, %)", -50, 50, -5, step=1)

    left, right = st.columns(2)
    with left:
        if st.button("Apply Scenario"):
            shock = pd.Series(0.0, index=px.columns, dtype=float)
            for k, v in scenario_lib[scn].items():
                if k in shock.index: shock.loc[k] = v
            pnl = float((w_opt * shock).sum()) * float(scale_live)
            st.success(f"[{scn}] Estimated instant P&L (post-scale): **{pnl:.2%}**")
    with right:
        if st.button("Apply Manual Shock"):
            if not pick:
                st.info("Bitte mindestens ein Asset auswählen.")
            else:
                shock = pd.Series(0.0, index=px.columns, dtype=float)
                shock.loc[pick] = shock_pct / 100.0
                pnl = float((w_opt * shock).sum()) * float(scale_live)
                friendly_list = ", ".join(labelize(pick))
                st.info(f"Manual shock on [{friendly_list}] P&L (post-scale): **{pnl:.2%}**")

# ---------- Trades / Rebalance Planner ----------
with tab_trades:
    st.subheader("Rebalance Planner")
    if "current_weights" not in st.session_state:
        st.session_state.current_weights = w_opt.copy()
    show_df = pd.DataFrame({"Current": st.session_state.current_weights.round(4)})
    show_df.index = labelize(list(show_df.index))
    edited = st.data_editor(show_df, use_container_width=True, key="cur_edit")
    current = edited["Current"]
    current.index = [ASSET_MAP.get(n, next((t for t,f in TICKER_TO_FRIENDLY.items() if f==n), n)) for n in current.index]
    current = current.reindex(w_opt.index).fillna(0.0)

    st.caption(f"Current sum: {current.sum():.3f}  ·  Target sum: {w_opt.sum():.3f}")
    if abs(current.sum() - 1.0) > 0.02:
        st.warning("Gewichtssumme ist nicht ~1.00 – bitte vor Trade planen normalisieren.")

    delta = (w_opt - current).rename("Delta (target-current)")
    last_prices = px.iloc[-1].reindex(w_opt.index).fillna(0.0)
    notional = (delta * float(portfolio_value)).rename("Trade Notional (EUR)")

    plan = pd.concat([
        current.rename("Current"),
        w_opt.rename("Target"),
        delta,
        last_prices.rename("Last Price"),
        notional
    ], axis=1).round(6)
    plan.index = labelize(list(plan.index))
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
    prices = px.iloc[-1]
    pos_val, equity = broker_mark_to_market(b, prices)
    c1,c2,c3 = st.columns(3)
    c1.metric("Cash", f"{b.cash:,.2f}"); c2.metric("Positions Value", f"{pos_val:,.2f}"); c3.metric("Equity", f"{equity:,.2f}")

    col1, col2, col3, col4 = st.columns([1.2,1,1,1])
    with col1:
        sym_friendly = st.selectbox("Symbol", labelize(list(px.columns)))
        sym = ASSET_MAP.get(sym_friendly, next((t for t,f in TICKER_TO_FRIENDLY.items() if f==sym_friendly), sym_friendly))
    with col2:
        side = st.selectbox("Side", ["BUY","SELL"])
    with col3:
        qty = st.number_input("Qty (shares)", min_value=0.0, value=0.0, step=1.0)
    with col4:
        if st.button("Submit Order"):
            px_now = prices.get(sym, np.nan)
            if np.isnan(px_now) or qty<=0:
                st.warning("Ungültiger Preis oder Menge.")
            else:
                cost = qty * px_now
                if side=="BUY":
                    if b.cash >= cost:
                        b.cash -= cost; b.positions[sym] = b.positions.get(sym,0.0) + qty
                        b.history.append({"ts":datetime.now().isoformat(),"sym":sym,"side":side,"qty":qty,"px":px_now})
                        st.success("Order ausgeführt.")
                    else:
                        st.error("Nicht genug Cash.")
                else:
                    have = b.positions.get(sym,0.0)
                    if have >= qty:
                        b.positions[sym] = have - qty; b.cash += cost
                        b.history.append({"ts":datetime.now().isoformat(),"sym":sym,"side":side,"qty":qty,"px":px_now})
                        st.success("Order ausgeführt.")
                    else:
                        st.error("Nicht genug Bestand.")

    st.markdown("**Open Positions**")
    if b.positions:
        df_pos = pd.DataFrame.from_dict(b.positions, orient="index", columns=["Shares"])
        df_pos["Last Price"] = [prices.get(k,np.nan) for k in df_pos.index]
        df_pos["Market Value"] = df_pos["Shares"] * df_pos["Last Price"]
        df_pos_disp = df_pos.round(4)
        df_pos_disp.index = labelize(list(df_pos_disp.index))
        st.dataframe(df_pos_disp, use_container_width=True)
    else:
        st.write("Keine Positionen.")

    st.markdown("**Order History**")
    if b.history:
        hist = pd.DataFrame(b.history)
        if not hist.empty:
            hist["sym"] = hist["sym"].map(lambda t: TICKER_TO_FRIENDLY.get(t,t))
        st.dataframe(hist, use_container_width=True)
    else:
        st.write("Keine Orders bisher.")

# ---------- AI Insights ----------
with tab_ai:
    st.subheader("AI Insights (Explain & Suggest)")
    notes = []
    crypto_assets = [c for c in w_opt.index if c in CRYPTO_TICKERS]
    crypto_exp = float(pd.Series(w_opt.loc[crypto_assets]).clip(lower=0).sum()) if crypto_assets else 0.0
    top_w = w_opt.sort_values(ascending=False).head(3)
    notes.append("Top Weights: " + ", ".join([f"{TICKER_TO_FRIENDLY.get(c,c)} {v:.0%}" for c,v in top_w.items()]) + ".")
    notes.append(f"Crypto Exposure: {crypto_exp:.0%} (Cap {crypto_cap:.0%}).")
    notes.append(f"Regime detected: **{regime}**; risk model: {('EWMA' if use_ewma_now else 'Ledoit-Wolf')}.")
    notes.append(f"Vol target {target_vol:.0%} → live leverage: {scale_live:.2f}×.")
    notes.append(f"Risk snapshot → VaR1d: {VaR_eff:.2%}, ES1d: {ES_eff:.2%}, MaxDD: {MDD_eff:.2%}.")
    if crypto_exp > crypto_cap + 1e-6: notes.append("🔧 Reduce crypto weights to respect cap.")
    if MDD_eff < -0.25: notes.append("🛡 Drawdown heavy: lower target vol or add hedges (Gold / MSCI World / long bonds).")
    if sr_eff < 0.6 and ann_vol_eff > target_vol * 0.9: notes.append("⚖️ Low Sharpe w/ high vol → try HRP or Min-CVaR + tune momentum tilt.")
    try:
        from sklearn.cluster import KMeans
        feat = pd.DataFrame({
            "vol": rets.rolling(20).std().iloc[-90:].mean(),
            "mom": (px.iloc[-1]/px.iloc[-63]-1.0).reindex(rets.columns)
        }).fillna(0.0)
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(feat.values)
        cl = km.labels_; high_idx = np.argmax([feat.values[cl==i,0].mean() for i in [0,1]])
        high_cluster_members = [TICKER_TO_FRIENDLY.get(a,a) for a in feat.index[cl==high_idx].tolist()]
        notes.append("🤖 ML Regime: High-vol cluster → " + ", ".join(high_cluster_members[:6]) + ("…" if len(high_cluster_members)>6 else ""))
    except Exception:
        notes.append("🤖 ML Regime: (optional) install scikit-learn for clustering insights.")
    st.write("\n\n".join(f"- {m}" for m in notes))

# ---------- Report ----------
with tab_report:
    st.subheader("One-Click HTML Report")
    weights_named = {TICKER_TO_FRIENDLY.get(k,k): float(v) for k,v in w_opt.round(4).to_dict().items()}
    html = f"""
    <html><head><meta charset="utf-8"><title>ALADDIN 3.1 Report</title></head>
    <body style="font-family:Inter,system-ui,sans-serif;background:#0c0f14;color:#e5e7eb;">
      <h2>ALADDIN 3.1 — Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})</h2>
      <p><b>Universe:</b> {', '.join(friendly_selection)}</p>
      <p><b>Optimizer:</b> {opt_label} | <b>Regime:</b> {regime} | <b>Caps:</b> Single {single_cap:.0%} / Equities {equity_cap:.0%} / Crypto {crypto_cap:.0%} | <b>Leverage:</b> {scale_live:.2f}×</p>
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
      <p style="color:#94a3b8;">Hinweis: BL/HRP/Min-CVaR benötigen optionale Libraries; bei Fehlen wird ein robustes Fallback genutzt.</p>
    </body></html>
    """
    st.download_button("Download HTML Report", data=html.encode("utf-8"),
                       file_name="aladdin_report.html", mime="text/html")
# ================== END ==================
