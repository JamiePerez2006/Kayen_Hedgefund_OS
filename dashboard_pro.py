# ==========================================================
# Alladdin 2.0 — MAXI HEDGEFUND (Single-file Streamlit App)
# Premium Look • Robust Backtest • Risk • Factors • Stress • Report
# ==========================================================

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.pyplot as plt

# ---------- Page / Theme ----------
st.set_page_config(page_title="Alladdin 2.0 — MAXI HEDGEFUND", layout="wide")

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
.block-container { padding-top: 1.0rem; padding-bottom: 1.4rem; }
h1,h2,h3 { letter-spacing:.2px; }
.kpi-card { background:rgba(22,26,35,.80); border:1px solid rgba(255,255,255,.06);
  border-radius:14px; padding:16px 18px; box-shadow:0 10px 28px rgba(0,0,0,.28); }
.kpi-big { font-size:2.0rem; font-weight:800; color:#ecfeff; line-height:1.1; }
.kpi-label { color:#9ca3af; font-size:.92rem; }
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent); }
.smallnote { color:#9ca3af; font-size:.9rem; }
.stDataFrame { border:1px solid rgba(255,255,255,.06); border-radius:12px; overflow:hidden; }
.stButton>button { border-radius:10px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Constants ----------
CRYPTO_KEYS = ("BTC", "ETH", "SOL", "DOGE", "ADA")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=300)
def load_prices(tickers, years=5):
    """Download Adjusted Close; always return DataFrame with columns=tickers."""
    period = f"{years}y"
    data = yf.download(
        tickers, period=period, auto_adjust=True, progress=False,
        group_by="ticker", threads=True
    )

    # Normalize to single-level columns
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        lvl1 = data.columns.get_level_values(1)
        px = data.xs("Adj Close", axis=1, level=1) if "Adj Close" in lvl1 else data.xs("Close", axis=1, level=1)
    else:
        px = data if isinstance(data, pd.DataFrame) else data.to_frame()

    if isinstance(px, pd.Series):
        px = px.to_frame()

    cols = [t for t in tickers if t in px.columns]
    px = px.reindex(columns=cols).dropna(how="all")
    return px

def to_returns(px, log=False):
    px = px.sort_index()
    rets = np.log(px/px.shift(1)) if log else px.pct_change()
    return rets.dropna(how="all")

def annualize_stats(series, freq=252):
    if len(series) == 0: return 0.0, 0.0
    mu  = float(series.mean()) * freq
    vol = float(series.std()) * math.sqrt(freq)
    return mu, vol

def sharpe_ratio(series, rf=0.0, freq=252):
    mu, vol = annualize_stats(series - rf/freq, freq=freq)
    return 0.0 if vol == 0 else mu/vol

def hist_var_es(series, alpha=0.95):
    s = series.dropna().values
    if s.size == 0: return 0.0, 0.0
    s = np.sort(s)
    k = int((1.0 - alpha) * len(s))
    k = max(0, min(len(s)-1, k))
    var = s[k]
    es  = s[:k+1].mean() if k >= 0 else var
    return float(var), float(es)

def max_drawdown(cum):
    if len(cum) == 0: return 0.0
    dd = (cum / cum.cummax()) - 1.0
    return float(dd.min())

def try_min_vol_weights(px_hist, min_w=0.0, max_w=0.35):
    """
    Try PyPortfolioOpt min volatility; fallback to heuristic.
    Returns (weights: pd.Series, used_optimizer: bool)
    """
    try:
        from pypfopt.risk_models import CovarianceShrinkage
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.efficient_frontier import EfficientFrontier

        mu = mean_historical_return(px_hist)
        S  = CovarianceShrinkage(px_hist).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
        w  = pd.Series(ef.min_volatility(), dtype=float)
        w.index = pd.Index(w.index, dtype=str)
        w = w.fillna(0.0)
        if abs(w.sum() - 1.0) > 1e-9:
            w /= w.sum()
        return w, True
    except Exception:
        return min_vol_heuristic(px_hist, min_w=min_w, max_w=max_w), False

def _project_to_simplex_with_bounds(w, min_w, max_w):
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= s
        w = np.clip(w, min_w, max_w)
        w /= w.sum()
    return w

def min_vol_heuristic(px_hist, min_w=0.0, max_w=0.35):
    """Inverse-Varianz Heuristik, robust & schnell (keine Solver)."""
    px_hist = px_hist.dropna()
    n = px_hist.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    if px_hist.shape[0] < 40:
        return pd.Series(np.ones(n)/n, index=px_hist.columns, dtype=float)
    rets = to_returns(px_hist)
    var = rets.var().replace([np.inf, -np.inf], np.nan).fillna(1.0)
    iv  = 1.0 / (var + 1e-12)
    w   = iv.values
    w   = _project_to_simplex_with_bounds(w, min_w, max_w)
    return pd.Series(w, index=px_hist.columns, dtype=float)

def apply_crypto_cap(weights: pd.Series, crypto_cap: float):
    """Kappt Summe der Krypto-Gewichte und renormalisiert den Rest."""
    w = weights.copy().fillna(0.0).astype(float)
    crypto = [c for c in w.index if any(k in c for k in CRYPTO_KEYS)]
    if crypto_cap <= 0:
        # Nur Non-Crypto behalten & renormieren
        nca = [c for c in w.index if c not in crypto]
        if nca:
            w.loc[crypto] = 0.0
            s = w.loc[nca].sum()
            w.loc[nca] = (w.loc[nca] / s) if s > 0 else (1.0/len(nca))
        return w

    if not crypto: return w
    cs = float(w.loc[crypto].sum())
    if cs > crypto_cap and cs > 0:
        scale = crypto_cap / cs
        w.loc[crypto] = w.loc[crypto] * scale
        nca = [c for c in w.index if c not in crypto]
        rem = 1.0 - w.sum()
        if nca and rem > 0:
            denom = float(w.loc[nca].sum())
            if denom <= 0:
                w.loc[nca] += rem / len(nca)
            else:
                w.loc[nca] += rem * (w.loc[nca] / denom)
    return w

# ---------- Sidebar ----------
st.title("Alladdin 2.0 — MAXI HEDGEFUND")
st.caption(datetime.now().strftime("%d.%m.%Y %H:%M"))

st.sidebar.header("Settings")
preset = st.sidebar.checkbox("Use KAYEN preset universe", value=True)
default_tickers = ["EURUSD=X", "SPY", "TLT", "GLD", "BTC-USD", "ETH-USD"]
choices = default_tickers + ["QQQ","ACWI","IEF","SLV","GDX","GLDM","XLK","XLF","XLE"]
tickers = st.sidebar.multiselect("Universe (tickers)", options=choices, default=default_tickers if preset else default_tickers)

years = st.sidebar.slider("Years of history", 2, 10, value=5)
min_w = st.sidebar.slider("Min weight per asset", 0.00, 0.20, value=0.00, step=0.01)
max_w = st.sidebar.slider("Max weight per asset", 0.10, 0.70, value=0.35, step=0.01)
crypto_cap = st.sidebar.slider("Crypto cap (total)", 0.00, 0.60, value=0.20, step=0.01)
alpha = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.90, 0.99, value=0.95, step=0.01)
portfolio_value = st.sidebar.number_input("Portfolio value (EUR)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")

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
w_opt, used_opt = try_min_vol_weights(px, min_w=min_w, max_w=max_w)
w_opt = w_opt.reindex(px.columns).fillna(0.0)
w_opt = apply_crypto_cap(w_opt, crypto_cap)
s = w_opt.sum()
w_opt = (w_opt / s) if s > 0 else pd.Series(np.ones(len(px.columns))/len(px.columns), index=px.columns, dtype=float)

# Current portfolio series
port = (rets.fillna(0.0) @ w_opt.values).rename("Port")
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
    st.caption(f"Optimizer: {'PyPortfolioOpt (min vol)' if used_opt else 'Inverse-Variance Heuristic'} · Crypto Cap: {crypto_cap:.0%}")

    L, R = st.columns([1.25, 1.0])
    with L:
        st.subheader("Portfolio Overview (target)")
        ytd = {t: rets[t].loc[rets.index.year == rets.index[-1].year].sum() if t in rets.columns else 0.0 for t in px.columns}
        df_over = pd.DataFrame({
            "Weight": w_opt.round(4),
            "YTD Return": pd.Series(ytd).round(4),
        })
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

    # Correlation Heatmap (matplotlib for crisp labels)
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

# ---------- Backtest (Monthly Rebalance) ----------
@st.cache_data(show_spinner=False, ttl=300)
def load_prices_bt(tickers, years):
    return load_prices(tickers, years=years)

def rebalance_backtest(px, rets, dates, min_w, max_w, crypto_cap, cost_bps):
    """
    Robust monthly rebalance with inverse-variance heuristic (fast).
    Applies crypto cap & transaction costs on turnover.
    Returns equity curve (Series).
    """
    w = None
    last_w = None
    eq = []
    equity = 1.0

    for dt in rets.index:
        if dt in dates:
            px_hist = px.loc[:dt].dropna()
            w_new = min_vol_heuristic(px_hist, min_w=min_w, max_w=max_w)
            # Crypto cap & normalize
            w_new = apply_crypto_cap(w_new, crypto_cap)
            s = w_new.sum()
            w_new = (w_new / s) if s > 0 else pd.Series(np.ones(len(w_new))/len(w_new), index=w_new.index)
            # Costs on turnover
            if last_w is not None:
                turnover = float((w_new - last_w).abs().sum())
                equity *= (1.0 - (cost_bps/10000.0) * turnover)
            w = w_new.copy()
            last_w = w.copy()

        if w is None:
            w = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)

        r = float(rets.loc[dt].reindex(w.index).fillna(0.0).dot(w.values))
        equity *= (1.0 + r)
        eq.append(equity)

    return pd.Series(eq, index=rets.index, name="Portfolio")

with tab_backtest:
    st.subheader("Backtest · Performance vs Benchmark  (Monthly Rebalance)")
    bt_years = st.slider("Backtest years", 2, 10, value=min(5, years), key="bt_years")
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0, 200, value=10, step=5, key="bt_cost_bps")
    bench_ticker = st.selectbox("Benchmark", ["SPY","ACWI","QQQ","IEF","GLD"], index=0, key="bench")

    px_bt = load_prices_bt(tickers, years=bt_years)
    rets_bt = to_returns(px_bt).dropna()
    rebal_dates = rets_bt.groupby([rets_bt.index.year, rets_bt.index.month]).head(1).index

    eq = rebalance_backtest(px_bt, rets_bt, rebal_dates, min_w=min_w, max_w=max_w, crypto_cap=crypto_cap, cost_bps=cost_bps)

    bench_px = yf.download(bench_ticker, period=f"{bt_years}y", auto_adjust=True, progress=False)["Close"].dropna()
    common = eq.index.intersection(bench_px.index)
    bench_eq = bench_px.loc[common] / bench_px.loc[common].iloc[0]
    port_eq  = eq.loc[common] / eq.loc[common].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=common, y=port_eq.values,  name="Portfolio",  mode="lines"))
    fig.add_trace(go.Scatter(x=common, y=bench_eq.values, name=bench_ticker, mode="lines"))
    fig.update_layout(title="Portfolio vs. Benchmark (Index=1.0)", xaxis_title="Date", yaxis_title="Index", height=420)
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

    outperf = (port_eq.iloc[-1]/bench_eq.iloc[-1]) - 1.0
    st.write(f"Outperformance vs {bench_ticker}: **{outperf:.2%}**")

# ---------- Factors (simple regression vs macro proxies) ----------
with tab_factors:
    st.subheader("Factor Attribution (Daily Returns Regression)")
    # simple macro factors: stocks, bonds, gold, USD proxy
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
        # fallback: least squares
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
    st.caption("Simuliere einen plötzlichen Schock auf ausgewählte Assets und sieh die unmittelbare P&L-Wirkung.")

    pick = st.multiselect("Assets to shock", options=list(px.columns), default=[px.columns[0]])
    shock_pct = st.slider("Shock size (one-day, %) — applied to selected assets", -30, 30, -5, step=1)
    if st.button("Apply Shock"):
        shock = pd.Series(0.0, index=px.columns, dtype=float)
        shock.loc[pick] = shock_pct / 100.0
        pnl = float((w_opt * shock).sum())
        st.success(f"Estimated instant portfolio P&L for shock {shock_pct:+d}% on {len(pick)} asset(s): **{pnl:.2%}**")

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
    <html><head><meta charset="utf-8"><title>Alladdin 2.0 Report</title></head>
    <body style="font-family:Inter,system-ui,sans-serif;background:#0e1117;color:#e5e7eb;">
      <h2>Alladdin 2.0 — Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})</h2>
      <p><b>Universe:</b> {', '.join(tickers)}</p>
      <p><b>Weights:</b> {w_opt.round(4).to_dict()}</p>
      <h3>Key Metrics</h3>
      <ul>
        <li>Annualized Return: {annualize_stats(port)[0]:.2%}</li>
        <li>Annualized Volatility: {annualize_stats(port)[1]:.2%}</li>
        <li>Sharpe: {sharpe_ratio(port):.2f}</li>
        <li>VaR({int(alpha*100)}%): {VaR:.2%}</li>
        <li>CVaR({int(alpha*100)}%) ~ {ES:.2%}</li>
        <li>Max Drawdown: {MDD:.2%}</li>
      </ul>
      <h3>Notes</h3>
      <p>Optimization: {'PyPortfolioOpt (min vol)' if used_opt else 'Inverse-Variance Heuristic'}; Crypto cap: {crypto_cap:.0%}.</p>
    </body></html>
    """
    st.download_button("Download HTML Report", data=html.encode("utf-8"),
                       file_name="alladdin_report.html", mime="text/html")

# ================== END ==================
