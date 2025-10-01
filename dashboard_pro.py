# Alladdin 2.0 – Maxi Hedgefund Dashboard (Pro)
# Vollständig eigenständig, robust & Streamlit-Cloud-tauglich

import math
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# -------------------------------- UI & Theme --------------------------------

st.set_page_config(layout="wide", page_title="MAXI Hedgefund — Pro")

# dezenter Premium-Look via Plotly Template + CSS
pio_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, Segoe UI, Roboto, sans-serif", size=14),
        paper_bgcolor="#111217",
        plot_bgcolor="#111217",
        colorway=["#22c55e", "#eab308", "#60a5fa", "#a78bfa", "#f97316", "#f43f5e", "#14b8a6"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False, showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=40, r=40, t=60, b=40),
    )
)
go.layout.template.default = pio_template

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; }
      h1,h2,h3 {letter-spacing: .2px;}
      .metric-row {display:flex; gap:18px; margin:12px 0 8px;}
      .card {flex:1; border:1px solid rgba(255,255,255,.07); background:#0f1116; border-radius:12px; padding:16px;}
      .card h4 {margin:0 0 8px 0; font-weight:600; color:#cbd5e1;}
      .big {font-size:24px; font-weight:700; color:#e5e7eb;}
      .muted {color:#94a3b8; font-size:12px;}
      .stDataFrame {border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,.06);}
      .css-1dp5vir, .stMarkdown {color:#dbe3ea;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ Helpers / Utils ------------------------------

CRYPTO_KEYS = ("BTC", "ETH", "SOL", "DOGE", "ADA")

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
    # Robust „Adj Close“ extrahieren → DataFrame mit Spalten = Ticker
    if isinstance(data, pd.DataFrame):
        if "Adj Close" in data.columns:
            px = data["Adj Close"].copy()
        elif isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(1):
                px = data.xs("Adj Close", axis=1, level=1)
            else:
                # notfalls „Close“
                px = data.xs("Close", axis=1, level=1)
        else:
            # evtl. direkt Close
            px = data
    else:
        px = data
    if isinstance(px, pd.Series):
        px = px.to_frame()
    # Nur gewünschte Ticker in gewünschter Reihenfolge
    cols = [t for t in tickers if t in px.columns]
    px = px.reindex(columns=cols).dropna(how="all")
    return px

def to_returns(px, log=False):
    px = px.sort_index()
    if log:
        rets = np.log(px / px.shift(1))
    else:
        rets = px.pct_change()
    return rets.dropna(how="all")

def annualize_stats(series, freq=252):
    mu = float(series.mean()) * freq if len(series) else 0.0
    vol = float(series.std()) * math.sqrt(freq) if len(series) else 0.0
    return mu, vol

def sharpe_ratio(series, rf=0.0, freq=252):
    mu, vol = annualize_stats(series-rf/freq, freq=freq)
    return 0.0 if vol == 0 else mu/vol

def hist_var_es(rets, alpha=0.95):
    if len(rets) == 0:
        return 0.0, 0.0
    arr = np.sort(rets.dropna().values)
    if len(arr) == 0:
        return 0.0, 0.0
    k = max(0, int((1 - alpha) * len(arr)) - 1)
    var = -arr[k] if k < len(arr) else -arr[-1]
    es = -arr[: k + 1].mean() if k >= 0 else 0.0
    return float(var), float(es)

def max_drawdown(cum):
    if len(cum) == 0:
        return 0.0
    dd = (cum / cum.cummax()) - 1.0
    return float(dd.min())

def _project_to_simplex_with_bounds(w, min_w, max_w):
    # clip to bounds, then renormalize; repeat to respect bounds
    w = np.clip(w, min_w, max_w)
    s = w.sum()
    if s <= 0:
        # gleich verteilen
        w[:] = 1.0 / len(w)
        return np.clip(w, min_w, max_w)
    w /= s
    # leichte Projektion, falls nach Clip Summe 1 nicht exakt
    w = np.clip(w, min_w, max_w)
    w /= w.sum()
    return w

def min_vol_heuristic(px_hist, min_w=0.0, max_w=0.35):
    # inverse-variance Heuristik, robust (keine externen Solver)
    px_hist = px_hist.dropna()
    if px_hist.shape[0] < 60:  # zu wenig Daten → equal weight
        n = px_hist.shape[1]
        return pd.Series(np.ones(n)/n, index=px_hist.columns, dtype=float)

    rets = to_returns(px_hist)
    iv = 1.0 / (rets.var() + 1e-12)
    iv = iv.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if iv.sum() <= 0:
        n = px_hist.shape[1]
        return pd.Series(np.ones(n)/n, index=px_hist.columns, dtype=float)

    w = iv.values
    w = _project_to_simplex_with_bounds(w, min_w, max_w)
    return pd.Series(w, index=px_hist.columns, dtype=float)

def apply_crypto_cap(w, crypto_cap):
    if crypto_cap <= 0:
        # alles auf Non-Crypto projizieren
        non_crypto = [c for c in w.index if not any(k in c for k in CRYPTO_KEYS)]
        if len(non_crypto) == 0:
            return w  # nichts zu tun
        w_non = w.loc[non_crypto].clip(lower=0)
        if w_non.sum() > 0:
            w.loc[non_crypto] = w_non / w_non.sum()
            w.loc[[c for c in w.index if c not in non_crypto]] = 0.0
        return w

    crypto = [c for c in w.index if any(k in c for k in CRYPTO_KEYS)]
    if len(crypto) == 0:
        return w
    cs = float(w.loc[crypto].sum())
    if cs > crypto_cap:
        scale = crypto_cap / cs if cs > 0 else 0.0
        w.loc[crypto] *= scale
        # rest auf non-crypto normieren
        non_crypto = [c for c in w.index if c not in crypto]
        rem = 1.0 - w.sum()
        if len(non_crypto) > 0 and rem > 1e-9:
            add = (w.loc[non_crypto] / max(w.loc[non_crypto].sum(), 1e-12)) * rem
            w.loc[non_crypto] += add
    return w

# ------------------------------- Sidebar / UI -------------------------------

st.title("MAXI HEDGEFUND — Pro")
st.caption(datetime.now().strftime("%d.%m.%Y %H:%M"))

st.sidebar.header("Settings")
preset = st.sidebar.checkbox("Use KAYEN preset universe", value=True)
default_tickers = ["EURUSD=X", "SPY", "TLT", "GLD", "BTC-USD", "ETH-USD"]
tickers = st.sidebar.multiselect(
    "Universe (tickers)",
    options=default_tickers if preset else default_tickers + ["QQQ","ACWI","IEF","SLV","GDX","GLDM"],
    default=default_tickers,
)
years = st.sidebar.slider("Years of history", 2, 10, value=5)
min_w = st.sidebar.slider("Min weight per asset", 0.00, 0.20, value=0.00, step=0.01)
max_w = st.sidebar.slider("Max weight per asset", 0.20, 0.60, value=0.35, step=0.01)
crypto_cap = st.sidebar.slider("Crypto cap (total)", 0.0, 0.50, value=0.20, step=0.01)
alpha = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.90, 0.99, value=0.95, step=0.01)
portfolio_value = st.sidebar.number_input("Portfolio value (EUR)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")

if len(tickers) == 0:
    st.warning("Bitte wähle mindestens einen Ticker aus.")
    st.stop()

# ------------------------------- Data Loading -------------------------------

try:
    px = load_prices(tickers, years=years)
except Exception as e:
    st.error(f"Konnte keine Daten laden: {e}")
    st.stop()

rets = to_returns(px).dropna(how="all")

# -------------------------- Optimierung + Crypto Cap ------------------------

weights = min_vol_heuristic(px, min_w=min_w, max_w=max_w)
weights = weights.fillna(0.0)
weights = apply_crypto_cap(weights.copy(), crypto_cap=crypto_cap).clip(lower=0.0)
weights = (weights / max(weights.sum(), 1e-12)).clip(lower=0.0)

# Portfolio-Serie
port = (rets.fillna(0.0) @ weights.values).rename("Port")
cum = (1.0 + port).cumprod()

ann_ret, ann_vol = annualize_stats(port)
sharpe = sharpe_ratio(port)
var, es = hist_var_es(port, alpha=alpha)
mdd = max_drawdown(cum)

# Rolling Flags / einfache Warnungen
rolling_sharpe = port.rolling(60).apply(lambda x: 0.0 if np.std(x)==0 else (np.mean(x)/np.std(x))*np.sqrt(252), raw=True)
rolling_dd = (cum / cum.cummax() - 1.0).rolling(252).min()

# ---------------------------------- LAYOUT ----------------------------------

tab_overview, tab_risk, tab_backtest, tab_trades = st.tabs(["Overview", "Risk", "Backtest", "Trades"])

with tab_overview:
    # KPI Cards
    st.subheader("MAXI HEDGEFUND DASHBOARD — Pro")
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h4>Portfolio Index</h4><div class="big">{cum.iloc[-1]:.2f}x</div><div class="muted">since start</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h4>Sharpe</h4><div class="big">{sharpe:.2f}</div><div class="muted">annualized</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h4>VaR({int(alpha*100)}%)</h4><div class="big">{var:.2%}</div><div class="muted">1-day</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h4>Max Drawdown</h4><div class="big">{mdd:.2%}</div><div class="muted">since start</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write(f"Optimizer used: **Min-Variance heuristic** (no solver).")
    if any(k in " ".join(tickers) for k in CRYPTO_KEYS):
        st.write(f"Crypto exposure: **{weights[[c for c in weights.index if any(k in c for k in CRYPTO_KEYS)]].sum():.2%}** (cap {crypto_cap:.0%})")

    # Zielgewichte & Risiko-Metriken
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.subheader("Portfolio Overview (target)")
        table = pd.DataFrame({
            "Weight": weights.round(4),
            "YTD Return": [rets.loc[rets.index.year == rets.index[-1].year, t].sum() if t in rets.columns else 0.0 for t in weights.index],
            "Risk Share": ( (rets.cov().values @ weights.values) * weights.values ).ravel()
        }, index=weights.index).round(4)
        st.dataframe(table, use_container_width=True)
    with c2:
        st.subheader("Risk Metrics")
        risk_df = pd.DataFrame({
            "Annualized Return": [ann_ret],
            "Annualized Volatility": [ann_vol],
            "Sharpe": [sharpe],
            f"VaR({int(alpha*100)}%)": [var],
            f"CVaR({int(alpha*100)}%) (approx)": [es],
            "Max Drawdown": [mdd],
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

with tab_risk:
    # Drawdown Chart & Rolling Sharpe
    st.subheader("Drawdowns & Rolling Sharpe")
    fig = go.Figure()
    dd = cum / cum.cummax() - 1.0
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines"))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name="Rolling Sharpe(60d)", mode="lines"))
    st.plotly_chart(fig_rs, use_container_width=True)

with tab_backtest:
    st.subheader("Backtest · Performance vs Benchmark  (Monthly Rebalance)")
    bt_years = st.slider("Backtest years", 2, 10, value=min(5, years), key="bt_years")
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0, 200, value=10, step=5, key="bt_cost_bps")
    bench_ticker = st.selectbox("Benchmark", ["SPY", "ACWI", "QQQ", "IEF", "GLD"], index=0, key="bench")

    @st.cache_data(show_spinner=False, ttl=300)
    def load_bench(ticker, years):
        bpx = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)["Close"].dropna()
        return bpx

    # lokale Kopien, damit Cache-Hash stabil bleibt
    px_bt  = load_prices(tickers, years=bt_years)
    rets_bt = to_returns(px_bt).dropna()
    # Rebalance-Termine: 1. Handelstag je Monat
    rebal_dates = rets_bt.groupby([rets_bt.index.year, rets_bt.index.month]).head(1).index

    def rebalance_backtest(px, rets, dates, min_w, max_w, crypto_cap, cost_bps):
        # w & equity
        w = None
        last_w = None
        equity = 1.0
        eq = []

        for dt in rets.index:
            # Neu gewichten an Rebalance-Tagen
            if dt in dates:
                px_hist = px.loc[:dt].dropna()
                w_new = min_vol_heuristic(px_hist, min_w=min_w, max_w=max_w)

                # sicher als Series mit Index
                if not isinstance(w_new, pd.Series):
                    w_new = pd.Series(np.asarray(w_new).ravel(), index=px_hist.columns, dtype=float)
                else:
                    w_new = w_new.reindex(px_hist.columns).astype(float)

                # Crypto-Kappung
                w_new = apply_crypto_cap(w_new.copy(), crypto_cap=crypto_cap).clip(lower=0.0)
                s = w_new.sum()
                w_new = (w_new / s) if s > 0 else pd.Series(np.ones(len(w_new)) / len(w_new), index=w_new.index)

                # Kosten auf Turnover
                if last_w is not None:
                    turnover = float((w_new - last_w).abs().sum())
                    equity *= 1.0 - (cost_bps / 10000.0) * turnover

                w = w_new.copy()
                last_w = w_new.copy()

            # Tages-Performance mit aktuellen Gewichten
            if w is None:
                # Falls ganz am Anfang noch kein Rebalance: equal weight
                w = pd.Series(1.0 / len(rets.columns), index=rets.columns, dtype=float)

            r = float(rets.loc[dt].reindex(w.index).fillna(0.0).dot(w.values))
            equity *= (1.0 + r)
            eq.append(equity)

        return pd.Series(eq, index=rets.index, name="Portfolio")

    eq = rebalance_backtest(px_bt, rets_bt, rebal_dates, min_w=min_w, max_w=max_w, crypto_cap=crypto_cap, cost_bps=cost_bps)

    # Benchmark laden & auf Backtest-Zeitraum matchen
    bench_px = load_bench(bench_ticker, bt_years)
    common_idx = eq.index.intersection(bench_px.index)
    bench_eq = (bench_px.loc[common_idx] / bench_px.loc[common_idx].iloc[0]).rename("Benchmark")
    port_eq  = (eq.loc[common_idx] / eq.loc[common_idx].iloc[0]).rename("Portfolio")

    # Chart
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=common_idx, y=port_eq.values, name="Portfolio", mode="lines"))
    fig_bt.add_trace(go.Scatter(x=common_idx, y=bench_eq.values, name=bench_ticker, mode="lines"))
    fig_bt.update_layout(title="Portfolio vs. Benchmark (Index=1.0)", xaxis_title="Date", yaxis_title="Index", height=420)
    st.plotly_chart(fig_bt, use_container_width=True)

    # Kennzahlen
    bench_ret = bench_eq.pct_change().dropna()
    b_cagr, b_vol = annualize_stats(bench_ret)
    b_sharpe = 0.0 if bench_ret.std()==0 else (bench_ret.mean()/bench_ret.std())*np.sqrt(252)
    b_mdd = max_drawdown(bench_eq)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bench CAGR", f"{b_cagr:.2%}")
    c2.metric("Bench Vol", f"{b_vol:.2%}")
    c3.metric("Bench Sharpe", f"{b_sharpe:.2f}")
    c4.metric("Bench MaxDD", f"{b_mdd:.2%}")

    outperf = float(port_eq.iloc[-1] / bench_eq.iloc[-1]) - 1.0
    st.write(f"Outperformance vs {bench_ticker} (letzter Stand): **{outperf:.2%}**")

with tab_trades:
    st.subheader("Target Weights & Rebalance Plan")
    # „momentane“ Zielgewichte editierbar (wird im Browserzustand gehalten)
    if "current_weights" not in st.session_state:
        st.session_state.current_weights = weights.copy()

    current_df = pd.DataFrame({"Current Weight": st.session_state.current_weights.round(4)})
    edited = st.data_editor(current_df, use_container_width=True)
    current = edited["Current Weight"].reindex(weights.index).fillna(0.0)

    st.caption(f"Current weights sum: {current.sum():.2f} (should be ~1.00)")
    if abs(current.sum() - 1.0) > 0.02:
        st.warning("Summe weicht > 2% von 1.00 ab – normalize before trading!")

    delta = (weights - current).rename("Delta (target – current)")
    prices = px.iloc[-1].reindex(weights.index).fillna(0.0)
    trade_notional = (delta * float(portfolio_value)).rename("Trade Notional (EUR)")
    plan = pd.concat([current.rename("Current"), weights.rename("Target"), delta, prices.rename("Last Price"), trade_notional], axis=1).round(4)
    st.dataframe(plan, use_container_width=True)

    # CSV-Download
    buf = io.BytesIO()
    plan.round(6).to_csv(buf, index=True)
    st.download_button("Download Rebalance Plan (CSV)", data=buf.getvalue(), file_name="rebalance_plan.csv", mime="text/csv")

# -------------- Ende --------------
