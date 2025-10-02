# ==========================================================
# ALLADDIN 2.0 — MAXI HEDGEFUND (Single-file Streamlit App)
# Premium Look • Multi-Optimizer • Walk-Forward BT • EWMA/Regime
# Stress • Monte-Carlo • Rebalance Planner • Presets • Reports
# ==========================================================

import io
import os
import json
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt

# ---------- Page / Theme ----------
st.set_page_config(page_title="ALLADDIN 2.0 — MAXI HEDGEFUND", layout="wide")

pio.templates["alladdin"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, Segoe UI, Roboto, sans-serif", size=14),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        colorway=["#22d3ee", "#14b8a6", "#f59e0b", "#f43f5e", "#a78bfa", "#60a5fa"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
)
pio.templates.default = "alladdin"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
h1,h2,h3 { letter-spacing:.2px; }
.kpi-card { background:rgba(22,26,35,.85); border:1px solid rgba(255,255,255,.06);
  border-radius:14px; padding:16px 18px; box-shadow:0 10px 28px rgba(0,0,0,.28); }
.kpi-big { font-size:2.0rem; font-weight:800; color:#ecfeff; line-height:1.1; }
.kpi-label { color:#9ca3af; font-size:.92rem; }
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent); margin:10px 0 18px; }
.smallnote { color:#9ca3af; font-size:.9rem; }
.stDataFrame { border:1px solid rgba(255,255,255,.06); border-radius:12px; overflow:hidden; }
.stButton>button { border-radius:10px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.06); border-radius: 10px; padding: 8px 12px; }
.stTabs [aria-selected="true"] { background: rgba(34,211,238,.14); border:1px solid rgba(34,211,238,.35); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Constants ----------
CRYPTO_KEYS = ("BTC", "ETH", "SOL", "DOGE", "ADA")
DEFAULT_TICKERS = ["EURUSD=X", "SPY", "TLT", "GLD", "BTC-USD", "ETH-USD"]
MORE_CHOICES = DEFAULT_TICKERS + ["QQQ","ACWI","IEF","SLV","GDX","GLDM","XLK","XLF","XLE","UUP","^IRX","^TNX"]

# ---------- Helpers: math/risk ----------
def to_returns(px: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    px = px.sort_index()
    rets = np.log(px/px.shift(1)) if log else px.pct_change()
    return rets.dropna(how="all").fillna(0.0)

def annualize_stats(series: pd.Series, freq: int = 252) -> Tuple[float, float]:
    if series.empty: return 0.0, 0.0
    mu = float(series.mean()) * freq
    vol = float(series.std(ddof=0)) * math.sqrt(freq)
    return mu, vol

def sharpe_ratio(series: pd.Series, rf: float = 0.0, freq: int = 252) -> float:
    if series.empty: return 0.0
    excess = series - (rf / freq)
    vol = float(excess.std(ddof=0))
    return 0.0 if vol == 0 else float(excess.mean() / vol * math.sqrt(freq))

def hist_var_es(series: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    s = np.sort(series.dropna().values)
    if s.size == 0: return 0.0, 0.0
    k = max(0, int((1.0 - alpha) * len(s)) - 1)
    var = s[k]
    es = s[:k+1].mean() if k >= 0 else var
    return float(var), float(es)

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

def regime_tag(returns: pd.DataFrame, lookback: int = 21, thr: float = 0.02) -> str:
    if returns.empty: return "unknown"
    vol = returns.rolling(lookback).std().dropna().iloc[-1].mean()
    return "high-vol" if vol > thr else "normal"

# ---------- Data: robust loader with cache & light winsorize ----------
@st.cache_data(show_spinner=False, ttl=300)
def load_prices(tickers: List[str], years: int = 5) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=365*years)).strftime("%Y-%m-%d")
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    # Normalize columns to single index
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        px = df["Adj Close"].copy()
    elif isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        px = df.xs("Adj Close", axis=1, level=1) if "Adj Close" in lvl1 else df.xs("Close", axis=1, level=1)
    else:
        px = df if isinstance(df, pd.DataFrame) else df.to_frame()
    if isinstance(px, pd.Series):
        px = px.to_frame()
    # Keep requested columns order
    px = px.reindex(columns=[t for t in tickers if t in px.columns])
    # Light outlier fix on extreme single-day % moves > 40%
    pct = px.pct_change()
    bad = pct.abs() > 0.40
    if bad.any().any():
        px = px.copy()
        for col in px.columns:
            idx = bad[col][bad[col]].index
            for t in idx:
                prev = px[col].shift(1).get(t, np.nan)
                nxt = px[col].shift(-1).get(t, np.nan)
                rep = np.nanmean([prev, nxt])
                if not np.isnan(rep): px.at[t, col] = rep
    return px.dropna(how="all")

# ---------- Optimizers ----------
def _project_to_simplex_with_bounds(w: np.ndarray, lb: float, ub: float) -> np.ndarray:
    w = np.clip(w, lb, ub)
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / s
        w = np.clip(w, lb, ub)
        w = w / w.sum()
    return w

def inverse_variance_weights(px_hist: pd.DataFrame, lb: float, ub: float) -> pd.Series:
    n = px_hist.shape[1]
    if n == 0: return pd.Series(dtype=float)
    if px_hist.shape[0] < 40:
        return pd.Series(np.ones(n)/n, index=px_hist.columns, dtype=float)
    rets = to_returns(px_hist)
    var = rets.var().replace([np.inf, -np.inf], np.nan).fillna(1.0)
    iv = 1.0 / (var + 1e-12)
    w = _project_to_simplex_with_bounds(iv.values, lb, ub)
    return pd.Series(w, index=px_hist.columns, dtype=float)

def optimizer_min_vol(px_hist: pd.DataFrame, lb: float, ub: float, use_ewma: bool) -> Tuple[pd.Series, str]:
    # Try PyPortfolioOpt; fall back to inverse-variance
    try:
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.efficient_frontier import EfficientFrontier
        if use_ewma:
            S = ewma_cov(to_returns(px_hist), lam=0.94)
        else:
            from pypfopt.risk_models import CovarianceShrinkage
            S = CovarianceShrinkage(px_hist).ledoit_wolf()
        mu = mean_historical_return(px_hist)
        ef = EfficientFrontier(mu, S, weight_bounds=(lb, ub))
        w = pd.Series(ef.min_volatility())
        w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0)
        if abs(float(w.sum()) - 1.0) > 1e-9: w = w / w.sum()
        return w, ("Min-Vol (EWMA)" if use_ewma else "Min-Vol (Ledoit-Wolf)")
    except Exception:
        return inverse_variance_weights(px_hist, lb, ub), "Inverse-Variance"

def optimizer_hrp(px_hist: pd.DataFrame, lb: float, ub: float) -> Tuple[pd.Series, str]:
    try:
        from pypfopt.hierarchical_risk_parity import HRPOpt
        rets = to_returns(px_hist)
        hrp = HRPOpt(rets)
        w = pd.Series(hrp.optimize())
        w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0)
        if abs(float(w.sum()) - 1.0) > 1e-9: w = w / w.sum()
        # Enforce simple caps post-hoc
        w = pd.Series(_project_to_simplex_with_bounds(w.values, lb, ub), index=w.index)
        return w, "HRP"
    except Exception:
        return inverse_variance_weights(px_hist, lb, ub), "Inverse-Variance"

def optimizer_cvar(returns: pd.DataFrame, alpha: float, lb: float, ub: float, allow_short: bool) -> Tuple[pd.Series, str]:
    # Minimize CVaR with cvxpy; fallback to inverse-variance
    try:
        import cvxpy as cp
        R = returns.values  # T x N
        T, N = R.shape
        w = cp.Variable(N)
        z = cp.Variable(T)
        v = cp.Variable(1)   # VaR level
        tau = 1 - alpha
        obj = v + (1.0/(tau*T)) * cp.sum(z)

        cons = []
        if allow_short:
            cons += [cp.sum(cp.abs(w)) == 1.0, w >= -ub, w <= ub]
        else:
            cons += [cp.sum(w) == 1.0, w >= lb, w <= ub]
        cons += [z >= 0, z >= -(R @ w) - v]

        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
        w_opt = pd.Series(np.array(w.value).ravel(), index=returns.columns).fillna(0.0)
        # Cap & renorm
        w_opt = pd.Series(_project_to_simplex_with_bounds(w_opt.values, lb, ub), index=w_opt.index)
        return w_opt, "Min-CVaR"
    except Exception:
        return inverse_variance_weights(returns.cumsum(), lb, ub), "Inverse-Variance"

# ---------- Group/Crypto Caps ----------
def apply_crypto_cap(weights: pd.Series, crypto_cap: float) -> pd.Series:
    w = weights.copy().fillna(0.0).astype(float)
    crypto = [c for c in w.index if any(k in c for k in CRYPTO_KEYS)]
    if not crypto: return w
    cs = float(w.loc[crypto].clip(lower=0).sum())
    if crypto_cap <= 0:
        w.loc[crypto] = 0.0
        if w.sum() > 0: w = w / w.sum()
        return w
    if cs > crypto_cap and cs > 0:
        scale = crypto_cap / cs
        w.loc[crypto] = w.loc[crypto] * scale
        if w.sum() > 0: w = w / w.sum()
    return w

# ---------- UI: Sidebar ----------
st.title("ALLADDIN 2.0 — MAXI HEDGEFUND")
st.caption(datetime.now().strftime("%d.%m.%Y %H:%M"))

st.sidebar.header("Settings")
colA, colB = st.sidebar.columns([1,1])
with colA:
    if st.button("Clear data cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")
with colB:
    preset_toggle = st.checkbox("Use KAYEN preset", value=True)

tickers = st.sidebar.multiselect("Universe (tickers)", options=MORE_CHOICES, default=DEFAULT_TICKERS if preset_toggle else DEFAULT_TICKERS)
years = st.sidebar.slider("Years of history", 2, 15, value=5)
min_w = st.sidebar.slider("Min weight per asset", 0.00, 0.20, value=0.00, step=0.01)
max_w = st.sidebar.slider("Max weight per asset", 0.10, 0.70, value=0.35, step=0.01)
crypto_cap = st.sidebar.slider("Crypto cap (total)", 0.00, 0.80, value=0.20, step=0.01)
alpha = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.90, 0.99, value=0.95, step=0.01)
allow_short = st.sidebar.checkbox("Allow short (experimental)", value=False)
portfolio_value = st.sidebar.number_input("Portfolio value (EUR)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")

# Optimizer choice
opt_choice = st.sidebar.selectbox("Optimizer", ["Min-Vol", "HRP", "Min-CVaR (tail risk)"], index=0)
use_ewma = st.sidebar.checkbox("Use EWMA cov in high-vol regime (auto)", value=True)
# Walk-forward settings
wf_reb = st.sidebar.selectbox("Backtest rebalance", ["Monthly","Weekly"], index=0)
wf_cost_bps = st.sidebar.number_input("Transaction cost (bps per turnover)", 0, 200, value=10, step=5)
wf_turnover_cap = st.sidebar.slider("Turnover cap per rebalance", 0.05, 1.0, value=0.35, step=0.05)

# Presets: save/load
st.sidebar.markdown("**Presets**")
if st.sidebar.button("Save current preset"):
    preset = {
        "tickers": tickers, "years": years, "min_w": min_w, "max_w": max_w,
        "crypto_cap": crypto_cap, "alpha": alpha, "allow_short": allow_short,
        "portfolio_value": portfolio_value, "opt_choice": opt_choice,
        "use_ewma": use_ewma, "wf_reb": wf_reb, "wf_cost_bps": wf_cost_bps,
        "wf_turnover_cap": wf_turnover_cap
    }
    st.sidebar.download_button("Download preset.json", data=json.dumps(preset, indent=2).encode(),
                               file_name="preset.json", mime="application/json")
uploaded = st.sidebar.file_uploader("Load preset.json", type=["json"])
if uploaded:
    try:
        preset = json.load(uploaded)
        st.session_state.update(preset)
        st.sidebar.success("Preset loaded. Bitte Seite neu laden (R).")
    except Exception as e:
        st.sidebar.error(f"Preset error: {e}")

if len(tickers) == 0:
    st.warning("Bitte mindestens einen Ticker wählen.")
    st.stop()

# ---------- Data ----------
try:
    px = load_prices(tickers, years=years)
    if px.empty:
        st.error("Keine Preisdaten geladen—prüfe Ticker.")
        st.stop()
    rets = to_returns(px).dropna(how="all")
except Exception as e:
    st.error(f"Datenfehler: {e}")
    st.stop()

# ---------- Optimize (target) ----------
regime = regime_tag(rets)
use_ewma_now = (use_ewma and regime == "high-vol")

if opt_choice == "Min-Vol":
    w_opt, opt_label = optimizer_min_vol(px, lb=min_w, ub=max_w, use_ewma=use_ewma_now)
elif opt_choice == "HRP":
    w_opt, opt_label = optimizer_hrp(px, lb=min_w, ub=max_w)
else:  # Min-CVaR
    w_opt, opt_label = optimizer_cvar(rets, alpha=alpha, lb=min_w, ub=max_w, allow_short=allow_short)

w_opt = w_opt.reindex(px.columns).fillna(0.0)
w_opt = apply_crypto_cap(w_opt, crypto_cap)
if w_opt.sum() > 0: w_opt = w_opt / w_opt.sum()

# ---------- Portfolio series & risk ----------
port = (rets.fillna(0.0) @ w_opt.values).rename("Portfolio")
cum  = (1.0 + port).cumprod()
ann_ret, ann_vol = annualize_stats(port)
sr = sharpe_ratio(port)
VaR, ES = hist_var_es(port, alpha=alpha)
MDD = max_drawdown(cum)
roll_sharpe = port.rolling(60).apply(lambda x: 0.0 if np.std(x)==0 else (np.mean(x)/np.std(x))*np.sqrt(252), raw=True)
roll_dd = (cum/cum.cummax()-1.0)

# ---------- Tabs ----------
tab_overview, tab_risk, tab_backtest, tab_factors, tab_stress, tab_trades, tab_report = st.tabs(
    ["Overview", "Risk", "Backtest", "Factors", "Stress", "Trades", "Report"]
)

# ---------- Overview ----------
def kpi(label, value):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-big">{value}</div>
    </div>
    """, unsafe_allow_html=True)

with tab_overview:
    a,b,c,d = st.columns(4)
    with a: kpi("Portfolio Index", f"{cum.iloc[-1]:.2f}x")
    with b: kpi("Sharpe", f"{sr:.2f}")
    with c: kpi(f"VaR({int(alpha*100)}%) 1d", f"{VaR:.2%}")
    with d: kpi("Max Drawdown", f"{MDD:.2%}")
    st.caption(f"Optimizer: {opt_label} · Crypto Cap: {crypto_cap:.0%} · Regime: {regime}")

    L, R = st.columns([1.25, 1.0])
    with L:
        st.subheader("Portfolio Overview (target)")
        ytd = {t: rets[t].loc[rets.index.year == rets.index[-1].year].sum() if t in rets.columns else 0.0 for t in px.columns}
        df_over = pd.DataFrame({"Weight": w_opt.round(4), "YTD Return": pd.Series(ytd).round(4)})
        st.dataframe(df_over, use_container_width=True)
    with R:
        st.subheader("Risk Metrics")
        df_risk = pd.DataFrame({
            "Annualized Return": [ann_ret],
            "Annualized Volatility": [ann_vol],
            "Sharpe": [sr],
            f"VaR({int(alpha*100)}%)": [VaR],
            f"CVaR({int(alpha*100)}%) (approx)": [ES],
            "Max Drawdown": [MDD],
        }).round(4)
        st.dataframe(df_risk, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    corr = rets.corr()
    im = ax.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Risk ----------
with tab_risk:
    st.subheader("Drawdown & Rolling Sharpe")
    fig1 = go.Figure()
    dd = (cum/cum.cummax()-1.0)
    fig1.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines"))
    fig1.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="Rolling Sharpe (60d)", mode="lines"))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Monte Carlo VaR/ES (1d)")
    sims = st.slider("Simulations", 1000, 20000, 5000, step=1000)
    mu_vec = rets.mean().values
    cov = rets.cov().values
    try:
        rnd = np.random.multivariate_normal(mu_vec, cov, size=sims)
        port_sims = rnd @ w_opt.values
        VaR_mc = np.percentile(port_sims, (1-alpha)*100)
        ES_mc  = port_sims[port_sims<=VaR_mc].mean() if np.any(port_sims<=VaR_mc) else VaR_mc
        st.write(f"Monte Carlo VaR: **{VaR_mc:.2%}**,  ES: **{ES_mc:.2%}**")
    except Exception as e:
        st.warning(f"Monte Carlo nicht verfügbar: {e}")

# ---------- Backtest (Walk-Forward) ----------
def monthly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    g = pd.Series(index=index, data=True)
    return g.groupby([index.year, index.month]).head(1).index

def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    g = pd.Series(index=index, data=True)
    return g.groupby([index.year, index.isocalendar().week]).head(1).index

def rebalance_backtest(px: pd.DataFrame, rets: pd.DataFrame, dates: pd.DatetimeIndex,
                       lb: float, ub: float, crypto_cap: float, cost_bps: float,
                       turnover_cap: float, opt_choice: str, use_ewma: bool, alpha: float, allow_short: bool) -> pd.Series:
    w_prev = None
    eq = []
    equity = 1.0
    for dt in rets.index:
        if dt in dates:
            px_hist = px.loc[:dt].dropna()
            if opt_choice == "Min-Vol":
                w = optimizer_min_vol(px_hist, lb, ub, use_ewma)[0]
            elif opt_choice == "HRP":
                w = optimizer_hrp(px_hist, lb, ub)[0]
            else:
                w = optimizer_cvar(rets.loc[:dt].dropna(), alpha, lb, ub, allow_short)[0]

            w = apply_crypto_cap(w, crypto_cap)
            if w.sum() > 0: w = w / w.sum()

            if w_prev is not None:
                delta = (w - w_prev).abs().sum()
                if delta > turnover_cap:
                    w = w_prev + (w - w_prev) * (turnover_cap / delta)
                equity *= (1.0 - (cost_bps/10000.0) * delta)
            w_prev = w.copy()
        if w_prev is None:
            w_prev = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)
        r = float(rets.loc[dt].reindex(w_prev.index).fillna(0.0).dot(w_prev.values))
        equity *= (1.0 + r)
        eq.append(equity)
    return pd.Series(eq, index=rets.index, name="Portfolio")

with tab_backtest:
    st.subheader("Walk-Forward Backtest · Portfolio vs Benchmark")
    bt_years = st.slider("Backtest years", 2, 15, value=min(5, years), key="bt_years")
    px_bt = load_prices(tickers, years=bt_years)
    rets_bt = to_returns(px_bt).dropna()

    rebal_dates = monthly_dates(rets_bt.index) if wf_reb == "Monthly" else weekly_dates(rets_bt.index)
    eq = rebalance_backtest(px_bt, rets_bt, rebal_dates, lb=min_w, ub=max_w, crypto_cap=crypto_cap,
                            cost_bps=wf_cost_bps, turnover_cap=wf_turnover_cap,
                            opt_choice=opt_choice, use_ewma=use_ewma_now, alpha=alpha, allow_short=allow_short)

    bench_ticker = st.selectbox("Benchmark", ["SPY","ACWI","QQQ","IEF","GLD"], index=0, key="bench")
    bench_px = yf.download(bench_ticker, period=f"{bt_years}y", auto_adjust=True, progress=False)["Close"].dropna()
    common = eq.index.intersection(bench_px.index)
    bench_eq = bench_px.loc[common] / bench_px.loc[common].iloc[0]
    port_eq  = eq.loc[common] / eq.loc[common].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=common, y=port_eq.values,  name="Portfolio", mode="lines"))
    fig.add_trace(go.Scatter(x=common, y=bench_eq.values, name=bench_ticker, mode="lines"))
    fig.update_layout(title="Index (Start=1.0)", xaxis_title="Date", yaxis_title="Index", height=420)
    st.plotly_chart(fig, use_container_width=True)

    bt_ret = eq.pct_change().dropna()
    n = len(bt_ret) if len(bt_ret) else 1
    bt_cagr = (eq.iloc[-1] ** (252/n)) - 1 if len(eq) else 0.0
    bt_vol  = bt_ret.std() * np.sqrt(252) if len(bt_ret) else 0.0
    bt_sha  = (bt_ret.mean()/bt_ret.std()) * np.sqrt(252) if len(bt_ret) and bt_ret.std()!=0 else 0.0
    bt_mdd  = float((eq/eq.cummax() - 1.0).min()) if len(eq) else 0.0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Backtest CAGR", f"{bt_cagr:.2%}")
    c2.metric("Backtest Vol", f"{bt_vol:.2%}")
    c3.metric("Backtest Sharpe", f"{bt_sha:.2f}")
    c4.metric("Backtest MaxDD", f"{bt_mdd:.2%}")

    if len(port_eq) and len(bench_eq):
        outp = float(port_eq.iloc[-1] / bench_eq.iloc[-1] - 1.0)
        st.markdown(f"**Outperformance vs {bench_ticker}: {outp:.2%}**")
    else:
        st.markdown("Outperformance: n/a")

# ---------- Factors ----------
with tab_factors:
    st.subheader("Factor Attribution (Daily Returns Regression)")
    factor_map = {
        "Stocks (SPY)": "SPY",
        "Bonds (IEF)": "IEF",
        "Gold (GLD)": "GLD",
        "USD (UUP proxy)": "UUP",
    }
    fac_tickers = list(factor_map.values())
    fac_px = load_prices(fac_tickers, years=min(years, 5))
    fac_rets = to_returns(fac_px)

    y = port.reindex(fac_rets.index).dropna()
    X = fac_rets.reindex(y.index).fillna(0.0).copy()
    X.columns = list(factor_map.keys())

    beta = {}
    r2 = 0.0
    try:
        import statsmodels.api as sm
        X_ = sm.add_constant(X)
        model = sm.OLS(y.values, X_.values).fit()
        r2 = float(model.rsquared)
        coefs = dict(zip(["Const"] + list(X.columns), model.params))
        for k in X.columns: beta[k] = float(coefs.get(k, 0.0))
    except Exception:
        X_ = np.column_stack([np.ones(len(X)), X.values])
        coef, *_ = np.linalg.lstsq(X_, y.values, rcond=None)
        pred = X_ @ coef
        ss_res = np.sum((y.values - pred)**2)
        ss_tot = np.sum((y.values - y.values.mean())**2)
        r2 = 0.0 if ss_tot==0 else 1 - ss_res/ss_tot
        for i, k in enumerate(X.columns, start=1):
            beta[k] = float(coef[i])

    betas_df = pd.DataFrame({"Beta": beta}).T.round(4)
    st.dataframe(betas_df, use_container_width=True)
    st.write(f"R² (explained variance): **{r2:.2f}**")

# ---------- Stress Testing ----------
with tab_stress:
    st.subheader("Instant Stress Test")
    st.caption("Simuliere einen 1-Tages-Schock auf Assets und sieh die unmittelbare P&L-Wirkung.")
    scenario_lib = {
        "Tech+Crypto Crash": {"AAPL": -0.15, "MSFT": -0.15, "SPY": -0.12, "BTC-USD": -0.25, "ETH-USD": -0.35},
        "Rates Spike": {"TLT": -0.08, "SPY": -0.04},
        "USD Surge": {"EURUSD=X": -0.02, "GLD": -0.03, "SPY": -0.03},
        "Crypto Winter": {"BTC-USD": -0.30, "ETH-USD": -0.40}
    }
    scn = st.selectbox("Scenario", list(scenario_lib.keys()), index=0)
    pick = st.multiselect("Assets to shock (optional override)", options=list(px.columns), default=[])
    shock_pct = st.slider("Shock size (manual, %)", -40, 40, -5, step=1)
    left, right = st.columns([1,1])
    with left:
        if st.button("Apply Scenario"):
            shock = pd.Series(0.0, index=px.columns, dtype=float)
            for k,v in scenario_lib[scn].items():
                if k in shock.index: shock.loc[k] = v
            pnl = float((w_opt * shock).sum())
            st.success(f"[{scn}] Estimated instant P&L: **{pnl:.2%}**")
    with right:
        if st.button("Apply Manual Shock"):
            shock = pd.Series(0.0, index=px.columns, dtype=float)
            for k in pick:
                shock.loc[k] = shock_pct / 100.0
            pnl = float((w_opt * shock).sum())
            st.info(f"Manual shock P&L: **{pnl:.2%}**")

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
    last_prices = px.iloc[-1].reindex(w_opt.index).fillna(0.0)
    notional = (delta * float(portfolio_value)).rename("Trade Notional (EUR)")

    plan = pd.concat([current.rename("Current"), w_opt.rename("Target"),
                      delta, last_prices.rename("Last Price"), notional], axis=1).round(6)
    st.dataframe(plan, use_container_width=True)

    buf = io.BytesIO()
    plan.to_csv(buf, index=True)
    st.download_button("Download Rebalance Plan (CSV)", data=buf.getvalue(),
                       file_name="rebalance_plan.csv", mime="text/csv")

# ---------- Report (HTML export) ----------
with tab_report:
    st.subheader("One-Click HTML Report")
    html = f"""
    <html><head><meta charset="utf-8"><title>ALLADDIN 2.0 Report</title></head>
    <body style="font-family:Inter,system-ui,sans-serif;background:#0e1117;color:#e5e7eb;">
      <h2>ALLADDIN 2.0 — Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})</h2>
      <p><b>Universe:</b> {', '.join(tickers)}</p>
      <p><b>Optimizer:</b> {opt_label} | <b>Regime:</b> {regime} | <b>Crypto cap:</b> {crypto_cap:.0%}</p>
      <h3>Weights</h3>
      <pre>{json.dumps({k: float(v) for k,v in w_opt.round(4).to_dict().items()}, indent=2)}</pre>
      <h3>Key Metrics</h3>
      <ul>
        <li>Annualized Return: {annualize_stats(port)[0]:.2%}</li>
        <li>Annualized Volatility: {annualize_stats(port)[1]:.2%}</li>
        <li>Sharpe: {sharpe_ratio(port):.2f}</li>
        <li>VaR({int(alpha*100)}%): {VaR:.2%}</li>
        <li>CVaR({int(alpha*100)}%) ~ {ES:.2%}</li>
        <li>Max Drawdown: {MDD:.2%}</li>
      </ul>
      <p class="smallnote">Hinweis: Min-CVaR/HRP benötigen optionale Libraries; bei Fehlen wird robustes Fallback genutzt.</p>
    </body></html>
    """
    st.download_button("Download HTML Report", data=html.encode("utf-8"),
                       file_name="alladdin_report.html", mime="text/html")

# ================== END ==================
