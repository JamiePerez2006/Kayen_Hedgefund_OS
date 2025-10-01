import io, math
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide", page_title="MAXI Hedgefund — Pro")

# ---- Plotly: einheitliches Template ----
import plotly.io as pio
import plotly.graph_objs as go

pio.templates["kayen"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=14),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        colorway=["#22d3ee", "#14b8a6", "#f59e0b", "#f43f5e", "#a78bfa"]
    )
)
pio.templates.default = "kayen"

# ---- Premium CSS ----
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 1.6rem; }
h1, h2, h3 { letter-spacing: .2px; }
.kpi-card { background:rgba(22,26,35,.78); border:1px solid rgba(255,255,255,.06);
            border-radius:14px; padding:16px 18px; box-shadow:0 10px 28px rgba(0,0,0,.28); }
.kpi-big { font-size:2.0rem; font-weight:800; color:#ecfeff; line-height:1.15; }
.kpi-label { color:#9ca3af; font-size:.92rem; }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,.10), transparent); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---- Header ----
c1, c2 = st.columns([0.80, 0.20])
with c1:
    st.markdown("## MAXI HEDGEFUND — Pro")
with c2:
    st.caption(datetime.now().strftime("%d.%m.%Y %H:%M"))
st.markdown("<hr/>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=300)
def load_prices(tickers, years=5):
    period = f"{years}y"
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # robustes Herausziehen der Adjusted Close Preise
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(1):
            px = data.xs("Adj Close", axis=1, level=1)
        else:
            px = data.xs("Close", axis=1, level=1)
    else:
        px = data
    if isinstance(px, pd.Series):
        px = px.to_frame()
    cols = [t for t in tickers if t in px.columns]
    px = px.reindex(columns=cols).dropna(how="all")
    if px.empty:
        raise ValueError("yfinance returned empty frame")
    return px

def to_returns(px, log=True):
    px = px.sort_index()
    rets = np.log(px/px.shift(1)) if log else px.pct_change()
    return rets.dropna(how="all").fillna(0.0)

def annualize_stats(rets, freq=252):
    mu  = rets.mean() * freq
    vol = rets.std(ddof=0) * np.sqrt(freq)
    return float(mu), float(vol)

def sharpe_ratio(rets, rf=0.0, freq=252):
    excess = rets - (rf/freq)
    vol = excess.std(ddof=0)
    return 0.0 if vol == 0 else float(excess.mean()/vol * np.sqrt(freq))

def hist_var_es(rets, alpha=0.95):
    arr = np.sort(rets.dropna().values)
    t   = max(1, int((1 - alpha) * len(arr)))
    var = -arr[t]
    es  = -arr[:t].mean() if t > 0 else var
    return float(var), float(es)

def max_drawdown(cum):
    dd = (cum / cum.cummax()) - 1.0
    return float(dd.min())

def try_min_vol_weights(px, min_w=0.0, max_w=0.35):
    """Versucht Min-Vol-Optimierung (PyPortfolioOpt). Fällt auf Equal-Weight zurück."""
    try:
        from pypfopt.risk_models import CovarianceShrinkage
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.efficient_frontier import EfficientFrontier
        mu = mean_historical_return(px)
        S  = CovarianceShrinkage(px).ledoit_wolf()
        ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
        ef.min_volatility()
        w = pd.Series(ef.clean_weights()).reindex(px.columns).fillna(0.0)
        s = w.abs().sum()
        if s > 0:
            w = w / s
        return w, True
    except Exception:
        n = len(px.columns)
        w = pd.Series(1/n, index=px.columns, dtype=float).clip(lower=min_w, upper=max_w)
        s = w.sum()
        w = w / s if s != 0 else pd.Series(1/n, index=px.columns, dtype=float)
        return w, False

def ytd_return(px_series):
    if px_series.empty:
        return np.nan
    mask = px_series.index.year == px_series.index[-1].year
    year_prices = px_series[mask]
    if year_prices.empty:
        return np.nan
    return float(year_prices.iloc[-1]/year_prices.iloc[0] - 1.0)

def persistent_flag(series, threshold_func, lookback=10):
    flags = deque(maxlen=lookback); out = []
    for val in series:
        flags.append(bool(threshold_func(val)))
        out.append(len(flags) == lookback and all(flags))
    return pd.Series(out, index=series.index)

# ------------- UI -------------
st.set_page_config(page_title="Maxi Hedgefund Dashboard (Pro)", layout="wide")
st.title("MAXI HEDGEFUND DASHBOARD — Pro")
st.caption(datetime.now().strftime("%d.%m.%Y  %H:%M"))

st.sidebar.header("Settings")
preset = st.sidebar.checkbox("Use KAYEN preset universe", value=True)
default_tickers = ["EURUSD=X","SPY","TLT","GLD","BTC-USD","ETH-USD"] if preset else \
                  ["SPY","TLT","GLD","AAPL","MSFT","EURUSD=X","BTC-USD","ETH-USD"]
tickers = st.sidebar.multiselect("Universe (tickers)", default_tickers, default=default_tickers)
years   = st.sidebar.slider("Years of history", 2, 10, 5)
min_w   = st.sidebar.slider("Min weight per asset", 0.0, 0.2, 0.0, 0.01)
max_w   = st.sidebar.slider("Max weight per asset", 0.1, 1.0, 0.35, 0.01)
crypto_cap = st.sidebar.slider("Crypto cap (total)", 0.0, 1.0, 0.20, 0.01)
alpha   = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.80, 0.99, 0.95, 0.01)
portfolio_value = st.sidebar.number_input("Portfolio value (EUR)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")

if len(tickers) == 0:
    st.warning("Bitte wähle mindestens einen Ticker aus.")
    st.stop()

# Daten laden
try:
    px = load_prices(tickers, years=years)
except Exception as e:
    st.error(f"Konnte keine Daten laden: {e}")
    st.stop()

rets = to_returns(px, log=True).dropna()

# Optimierung + Crypto-Cap
weights, used_optimizer = try_min_vol_weights(px, min_w=min_w, max_w=max_w)
crypto_assets = [t for t in weights.index if any(sym in t for sym in ["BTC","ETH","SOL","DOGE","ADA"])]
crypto_sum = float(weights.reindex(crypto_assets).fillna(0.0).sum()) if crypto_assets else 0.0
if crypto_sum > crypto_cap:
    scale = crypto_cap / crypto_sum if crypto_sum > 0 else 0.0
    weights.loc[crypto_assets] *= scale
    non_crypto = [t for t in weights.index if t not in crypto_assets]
    rem = 1.0 - weights.sum()
    if non_crypto and rem > 0:
        weights.loc[non_crypto] += rem * (weights.loc[non_crypto] / weights.loc[non_crypto].sum())

# Portfolio-Stats
port = rets.dot(weights.values)
cum  = (1 + port).cumprod()
ann_ret, ann_vol = annualize_stats(port)
sharpe = sharpe_ratio(port)
var, es = hist_var_es(port, alpha=alpha)
mdd = max_drawdown(cum)

# Rolling + persistente Alerts
rolling_sharpe = port.rolling(60).apply(lambda x: (x.mean()/x.std())*np.sqrt(252) if np.std(x)!=0 else 0.0, raw=True)
rolling_cum = (1 + port).cumprod()
rolling_dd  = (rolling_cum/rolling_cum.cummax()-1.0)
sig_sharpe = persistent_flag(rolling_sharpe.dropna(), lambda v: v < 1.0, lookback=10)
roll = port.rolling(252)
hist_var = roll.apply(lambda x: -np.sort(x)[max(1, int(0.05*len(x)))], raw=True)
sig_var  = persistent_flag(hist_var.dropna(), lambda v: v > 0.02, lookback=10)
sig_dd   = persistent_flag(rolling_dd.dropna(),   lambda v: v < -0.15, lookback=10)

tab_overview, tab_risk, tab_backtest, tab_trades = st.tabs(["Overview", "Risk", "Backtest", "Trades"])

with tab_overview:
    st.subheader("Overview")

    # KPI-Karten
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Index", f"{cum.iloc[-1]:.2f}x")
    c2.metric("Sharpe", f"{sharpe:.2f}")
    c3.metric(f"VaR({int(alpha*100)}%)", f"{var:.2%}")
    c4.metric("Max Drawdown", f"{mdd:.2%}")
    st.write(f"Optimizer used: {'PyPortfolioOpt (min vol)' if used_optimizer else 'Fallback Equal-Weight'}")
    if crypto_assets:
        st.write(f"Crypto exposure: {crypto_sum:.2%}  (cap {crypto_cap:.0%})")
    
    # Tabellen
    lcol, rcol = st.columns([1.2, 1.0])
    with lcol:
        st.subheader("Portfolio Overview (target)")
        ytds = {t: ytd_return(px[t].dropna()) for t in weights.index if t in px.columns}
        cov = rets.cov()
        rc = (cov.values @ weights.values.reshape(-1,1) * weights.values.reshape(-1,1)).ravel()
        rc = pd.Series(rc, index=weights.index).clip(lower=0.0)
        table = pd.DataFrame({
            "Weight": weights.round(4),
            "YTD Return": pd.Series(ytds).round(4),
            "Risk Share": (rc/rc.sum()).round(4) if rc.sum()>0 else rc
        })
        st.dataframe(table, use_container_width=True)
    with rcol:
        st.subheader("Risk Metrics")
        risk_df = pd.DataFrame({
            "Annualized Return":[ann_ret],
            "Annualized Volatility":[ann_vol],
            "Sharpe":[sharpe],
            f"VaR({int(alpha*100)}%)":[var],
            f"CVaR({int(alpha*100)}%) (approx)":[es],
            "Max Drawdown":[mdd],
        }).round(4)
        st.dataframe(risk_df, use_container_width=True)
    
    # Heatmap
    st.subheader("Correlation Heatmap")
    fig1, ax1 = plt.subplots()
    corr = rets.corr()
    im = ax1.imshow(corr.values)
    ax1.set_xticks(range(len(corr.columns))); ax1.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax1.set_yticks(range(len(corr.index)));   ax1.set_yticklabels(corr.index)
    fig1.tight_layout()
    st.pyplot(fig1)

# ---------------- Backtest (Monthly Rebalance) ----------------
with tab_backtest:
    import plotly.graph_objects as go

    st.subheader("Backtest · Performance vs Benchmark")

    # Parameter für den Backtest
    bt_years = st.slider("Backtest years", 2, 10, years, key="bt_years")
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0, 200, 10, step=5, key="bt_cost_bps")

    @st.cache_data(show_spinner=False, ttl=300)
    def load_prices_bt(tickers, years):
        return load_prices(tickers, years=years)

    # Daten für Backtest laden
    px_bt = load_prices_bt(tickers, years=bt_years)
    rets_bt = to_returns(px_bt, log=True).dropna()

    # Rebalance-Termine: 1. Handelstag jedes Monats
    rebal_dates = rets_bt.groupby([rets_bt.index.year, rets_bt.index.month]).head(1).index

 # --- Backtest-Funktion: monatliche Rebalance, Kosten auf Turnover ---
def rebalance_backtest(px, rets, dates, min_w, max_w, cost_bps):
    w = None
    last_w = None
    eq = []
    equity = 1.0

    for dt in rets.index:
        # an Rebalance-Tagen neue Gewichte bestimmen
        if dt in dates:
            px_hist = px.loc[:dt].dropna()

            # Versuche Min-Vol-Gewichte
            w_new = try_min_vol_weights(px_hist, min_w=min_w, max_w=max_w)

            # --- ROBUSTER FALLBACK ---
            # Falls None/False oder falscher Typ -> Equal-Weight
            if not isinstance(w_new, pd.Series):
                w_new = pd.Series(1.0 / len(rets.columns), index=rets.columns)

            # Auf Returns-Spalten mappen und NAs füllen
            w_new = w_new.reindex(rets.columns).fillna(0.0)

            # Transaktionskosten (nur wenn es bereits alte Gewichte gab)
            if isinstance(last_w, pd.Series):
                turnover = float((w_new - last_w).abs().sum())
                equity *= (1.0 - (cost_bps / 10000.0) * turnover)

            # Gewichte übernehmen
            w = w_new
            last_w = w_new.copy()

        # Falls vor der ersten Rebalance noch keine Gewichte: Equal-Weight
        if w is None:
            w = pd.Series(1.0 / len(rets.columns), index=rets.columns)

        # Tages-Return mit aktuellen Gewichten (Index sicher ausrichten)
        r = float(rets.loc[dt].reindex(w.index).fillna(0.0).dot(w))
        equity *= (1.0 + r)
        eq.append(equity)

    return pd.Series(eq, index=rets.index, name="Portfolio")
    
    # Backtest laufen lassen
    eq = rebalance_backtest(px_bt, rets_bt, rebal_dates, min_w=min_w, max_w=max_w, cost_bps=cost_bps)

    # ---------------- Benchmark-Vergleich ----------------
    bench_ticker = st.selectbox("Benchmark", ["SPY","ACWI","QQQ","IEF","GLD"], index=0, key="benchf")

    @st.cache_data(show_spinner=False, ttl=300)
    def load_bench(ticker, years):
        bpx = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)["Close"].dropna()
        return bpx

    bench_px = load_bench(bench_ticker, bt_years)
    # auf gemeinsamen Zeitraum schneiden & auf 1.0 normieren
    common_idx = eq.index.intersection(bench_px.index)
    bench_eq = bench_px.loc[common_idx] / bench_px.loc[common_idx].iloc[0]
    port_eq  = eq.loc[common_idx] / eq.loc[common_idx].iloc[0]

    # Chart: Portfolio vs Benchmark (Index = 1.0)
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=common_idx, y=port_eq.values,  name="Portfolio", mode="lines"))
    fig_bt.add_trace(go.Scatter(x=common_idx, y=bench_eq.values, name=bench_ticker, mode="lines"))
    fig_bt.update_layout(title="Portfolio vs. Benchmark (Index=1.0)",
                         xaxis_title="Date", yaxis_title="Index",
                         margin=dict(l=0,r=0,t=40,b=0), height=420)
    st.plotly_chart(fig_bt, use_container_width=True)

    # Outperformance (Portfolio / Benchmark)
    outperf = (port_eq / bench_eq) - 1.0
    st.write(f"Outperformance vs {bench_ticker} (letzter Stand): {float(outperf.iloc[-1]):.2%}")
# ----------------------- Backtest: Benchmark & Interactive Chart -----------------------
import plotly.graph_objs as go

# Benchmark-Auswahl
bench_ticker = st.selectbox("Benchmark", ["SPY", "ACWI", "QQQ", "IEF", "GLD"], index=0, key="bench")

# Benchmark laden und auf denselben Zeitraum wie das Backtest-Equity (eq) mappen
try:
    bench_px = yf.download(
        bench_ticker,
        period=f"{bt_years}y",
        auto_adjust=True,
        progress=False
    )["Close"].dropna()

    # Benchmark auf Index 1 normieren und auf den eq-Index bringen
    bench_eq = (bench_px / bench_px.iloc[0]).reindex(eq.index).ffill().fillna(1.0)
except Exception:
    # Fallback: wenn Download/Mapping schiefgeht, konstant 1.0 als Benchmark
    bench_eq = pd.Series(1.0, index=eq.index, name="Benchmark")

# Kennzahlen (nur berechnen, wenn genügend Daten vorhanden sind)
eq_ret = eq.pct_change().dropna()
bench_ret = bench_eq.pct_change().dropna()

if len(eq_ret) >= 2 and len(bench_ret) >= 2:
    def ann_cagr(s):
        r = s.iloc[-1] / s.iloc[0]
        years = max((s.index[-1] - s.index[0]).days / 365.25, 1e-6)
        return r ** (1 / years) - 1

    p_cagr = ann_cagr(eq)
    b_cagr = ann_cagr(bench_eq)

    p_vol = eq_ret.std() * np.sqrt(252)
    b_vol = bench_ret.std() * np.sqrt(252)

    p_sharpe = (eq_ret.mean() / (p_vol / np.sqrt(252))) if p_vol != 0 else 0.0
    b_sharpe = (bench_ret.mean() / (b_vol / np.sqrt(252))) if b_vol != 0 else 0.0

    p_mdd = (eq / eq.cummax()).min() - 1
    b_mdd = (bench_eq / bench_eq.cummax()).min() - 1

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Bench CAGR", f"{b_cagr:.2%}")
    with c2: st.metric("Bench Vol", f"{b_vol:.2%}")
    with c3: st.metric("Bench Sharpe*", f"{b_sharpe:.2f}")
    with c4: st.metric("Bench MaxDD", f"{b_mdd:.2%}")

# Gemeinsames Plot-Fenster (Index=1 normiert)
common_idx = eq.index.union(bench_eq.index).unique().sort_values()
eq_plot = eq.reindex(common_idx).fillna(method="ffill").fillna(1.0)
bench_plot = bench_eq.reindex(common_idx).fillna(method="ffill").fillna(1.0)

fig_int = go.Figure()
fig_int.add_trace(go.Scatter(x=common_idx, y=eq_plot.values, name="Portfolio", mode="lines"))
fig_int.add_trace(go.Scatter(x=common_idx, y=bench_plot.values, name=bench_ticker, mode="lines"))
fig_int.update_layout(
    title="Portfolio vs. Benchmark (Index=1.0)",
    xaxis_title="Date",
    yaxis_title="Index",
    template="plotly_dark",
    hovermode="x unified",
)
st.plotly_chart(fig_int, use_container_width=True)

# --- Kennzahlen robust (immer auf Skalar casten) ---
bench_ret = bench_eq.pct_change().dropna()

b_len   = len(bench_ret) if len(bench_ret) > 0 else 1
b_end   = float(bench_eq.iloc[-1])
b_mean  = float(bench_ret.mean()) if len(bench_ret) else 0.0
b_std   = float(bench_ret.std())  if len(bench_ret) else 0.0

b_cagr  = (b_end ** (252 / b_len)) - 1
b_vol   = b_std * np.sqrt(252)
b_sharpe = (b_mean / b_std * np.sqrt(252)) if b_std != 0 else 0.0
b_mdd   = float((bench_eq / bench_eq.cummax() - 1.0).min())

cE, cF, cG, cH = st.columns(4)
cE.metric("Bench CAGR",   f"{b_cagr:.2%}")
cF.metric("Bench Vol",    f"{b_vol:.2%}")
cG.metric("Bench Sharpe", f"{b_sharpe:.2f}")
cH.metric("Bench MaxDD",  f"{b_mdd:.2%}")

# --- Interaktiver Chart: Portfolio vs. Benchmark ---
fig_int = go.Figure()
fig_int.add_trace(go.Scatter(x=eq.index,       y=eq.values,        name="Portfolio",    mode="lines"))
fig_int.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values,  name=bench_ticker,   mode="lines"))
fig_int.update_layout(title="Portfolio vs. Benchmark (Index=1.0)", xaxis_title="Date", yaxis_title="Index")
st.plotly_chart(fig_int, use_container_width=True)

# Gemeinsame Datenpunkte robust ausrichten
common_idx = eq.index.intersection(bench_eq.index).unique()
peq = eq.loc[common_idx]
beq = bench_eq.loc[common_idx]

outperf = (peq / beq) - 1.0
st.write(f"Outperformance vs {bench_ticker} (letzter Stand): {float(outperf.iloc[-1]):.2%}")

# Alerts (mit 10-Tage-Persistenz)
# --- Ensure 'current' weights exist and are normalized for Alerts ---
if "current" not in locals():
    if "current_weights" in st.session_state:
        current = pd.Series(st.session_state.current_weights, index=weights.index).fillna(0.0)
    else:
        # falls nichts im State ist: aktuelle = Zielgewichte als Fallback
        current = weights.copy()

current_sum = float(np.nansum(current.values))
if not np.isfinite(current_sum) or current_sum == 0:
    current_sum = 1.0
current = current / current_sum
st.subheader("Alerts")
alerts = []
if crypto_assets and crypto_sum > crypto_cap:
    alerts.append(f"Crypto exposure {crypto_sum:.1%} exceeds cap {crypto_cap:.0%}.")
if not sig_sharpe.empty and sig_sharpe.iloc[-1]:
    alerts.append("Sharpe < 1.0 for 10 consecutive days.")
if not sig_dd.empty and sig_dd.iloc[-1]:
    alerts.append("Max Drawdown < -15% for 10 consecutive days.")
if not sig_var.empty and sig_var.iloc[-1]:
    alerts.append("VaR(95%) > 2% (1d) for 10 consecutive days.")
if abs(current_sum - 1.0) > 0.02:
    alerts.append("Current weights do not sum to ~1.0 (normalize before trading).")
if len(alerts)==0:
    st.success("No alerts. All within limits.")
else:
    for a in alerts:
        st.warning(a)
