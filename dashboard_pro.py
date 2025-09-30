import io, math
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

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

# -------------------- Backtest (Monthly Rebalance) --------------------
st.subheader("Backtest (Monthly Rebalance)")
bt_years = st.slider("Backtest years", 2, 10, years, key="bt_years")
cost_bps = st.number_input("Transaction cost (bps per turnover)", 0, 200, 10, step=5, key="bt_cost_bps")

@st.cache_data(show_spinner=False, ttl=300)
def load_prices_bt(tickers, years):
    return load_prices(tickers, years=years)

px_bt  = load_prices_bt(tickers, years=bt_years)
rets_bt = to_returns(px_bt, log=True).dropna()

# Rebalance-Termine: 1. Handelstag des Monats
rebal_dates = rets_bt.groupby([rets_bt.index.year, rets_bt.index.month]).head(1).index

def rebalance_backtest(rets, dates, min_w, max_w, crypto_cap, cost_bps):
    w = None
    equity = [1.0]
    last_w = None
    for i, dt in enumerate(rets.index):
        if (i == 0) or (dt in dates):
            # Gewichte neu optimieren bis Stichtag dt
            px_hist = px_bt.loc[:dt].dropna()
            w, _ = try_min_vol_weights(px_hist, min_w=min_w, max_w=max_w)

            # Crypto-Kappung
            ca = [t for t in w.index if any(sym in t for sym in ["BTC","ETH","SOL","DOGE","ADA"])]
            cs = float(w.reindex(ca).fillna(0.0).sum()) if ca else 0.0
            if cs > crypto_cap:
                scale = crypto_cap / cs if cs > 0 else 0.0
                w.loc[ca] *= scale
                nc = [t for t in w.index if t not in ca]
                rem = 1.0 - w.sum()
                if nc and rem > 0:
                    w.loc[nc] += rem * (w.loc[nc] / w.loc[nc].sum())

            # Transaktionskosten auf Turnover (bps)
            if last_w is not None:
                turnover = float((w - last_w).abs().sum())
                equity[-1] *= (1.0 - (cost_bps / 10000.0) * turnover)

            last_w = w.copy()

        # Tagesrendite mit aktuellen Gewichten
        r = float(rets.loc[dt].reindex(w.index).fillna(0.0).dot(w.values))
        equity.append(equity[-1] * (1.0 + r))

    idx = pd.Index([rets.index[0]] + list(rets.index), name="Date")
    return pd.Series(equity, index=idx, name="Equity")

eq = rebalance_backtest(rets_bt, rebal_dates, min_w=min_w, max_w=max_w, crypto_cap=crypto_cap, cost_bps=cost_bps)
bt_ret = eq.pct_change().dropna()

bt_cagr   = (eq.iloc[-1])**(252/len(bt_ret)) - 1
bt_vol    = bt_ret.std() * np.sqrt(252)
bt_sharpe = bt_ret.mean()/bt_ret.std()*np.sqrt(252) if bt_ret.std()!=0 else 0.0
bt_mdd    = (eq/eq.cummax()-1).min()

cA, cB, cC, cD = st.columns(4)
cA.metric("Backtest CAGR",   f"{bt_cagr:.2%}")
cB.metric("Backtest Vol",    f"{bt_vol:.2%}")
cC.metric("Backtest Sharpe", f"{bt_sharpe:.2f}")
cD.metric("Backtest MaxDD",  f"{bt_mdd:.2%}")

fig_bt, ax_bt = plt.subplots()
ax_bt.plot(eq.index, eq.values)
ax_bt.set_title("Backtest Equity (Monthly Rebalance)")
fig_bt.tight_layout()
st.pyplot(fig_bt)

# -------------------- Backtest: Benchmark & Interactive Chart --------------------
import plotly.graph_objs as go

bench_ticker = st.selectbox("Benchmark", ["SPY", "ACWI", "QQQ", "IEF", "GLD"], index=0, key="bench")

# Benchmark laden und auf den Backtest-Zeitraum/Index mappen
bench_px = yf.download(
    bench_ticker, period=f"{bt_years}y", auto_adjust=True, progress=False
)["Close"].dropna()

bench_eq = (bench_px / bench_px.iloc[0])
# sicherstellen, dass eq existiert (kommt aus dem Backtest) und Index passt
bench_eq = bench_eq.reindex(eq.index).fillna(method="ffill").fillna(1.0)

# Falls trotzdem DataFrame entstanden ist: auf Series reduzieren
if isinstance(bench_eq, pd.DataFrame):
    bench_eq = bench_eq.iloc[:, 0]

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
