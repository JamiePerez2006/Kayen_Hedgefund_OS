# ==============================================
# MAXI HEDGEFUND — Pro (Full App, Premium Look)
# ==============================================

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# Plotly (interaktive Charts)
import plotly.io as pio
import plotly.graph_objs as go


# ---------------- Page & Premium Styling ----------------
st.set_page_config(page_title="MAXI Hedgefund — Pro", layout="wide")

# Plotly Template (einheitlicher Look)
pio.templates["kayen"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=14),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        colorway=["#22d3ee", "#14b8a6", "#f59e0b", "#f43f5e", "#a78bfa"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
)
pio.templates.default = "kayen"

# Premium CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
h1, h2, h3 { letter-spacing: .2px; }
.kpi-card { background:rgba(22,26,35,.80); border:1px solid rgba(255,255,255,.06);
            border-radius:14px; padding:16px 18px; box-shadow:0 10px 28px rgba(0,0,0,.28); }
.kpi-big { font-size:2.0rem; font-weight:800; color:#ecfeff; line-height:1.15; }
.kpi-label { color:#9ca3af; font-size:.92rem; }
hr { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.10),transparent); }
.smallnote { color:#9ca3af; font-size:.9rem; }
.stButton>button { border-radius:10px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
c1, c2 = st.columns([0.80, 0.20])
with c1:
    st.markdown("## MAXI HEDGEFUND — Pro")
with c2:
    st.caption(datetime.now().strftime("%d.%m.%Y %H:%M"))
st.markdown("<hr/>", unsafe_allow_html=True)


# ---------------- Helper Functions ----------------
@st.cache_data(show_spinner=False, ttl=300)
def load_prices(tickers, years=5):
    """Lade Adjusted Close für alle Ticker als DataFrame (Spalten=tickers)."""
    if isinstance(tickers, (list, tuple)):
        tickers = [t.strip() for t in tickers if str(t).strip()]
    else:
        tickers = [str(tickers).strip()]

    period = f"{years}y"
    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Robust MultiIndex/SingleIndex-Handling
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        px = data["Adj Close"].copy()
    elif isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        cols = data.columns
        px = data.xs("Adj Close", axis=1, level=1) if "Adj Close" in cols.get_level_values(1) else data.xs("Close", axis=1, level=1)
    else:
        px = data if isinstance(data, pd.DataFrame) else data.to_frame()

    cols_order = [t for t in tickers if t in px.columns]
    px = px.reindex(columns=cols_order).dropna(how="all")
    return px


def to_returns(px, log=True):
    px = px.sort_index()
    rets = np.log(px).diff() if log else px.pct_change()
    return rets.dropna(how="all")


def ann_stats(series, freq=252):
    if len(series) == 0:
        return 0.0, 0.0
    m = float(series.mean()); s = float(series.std())
    return m * freq, s * np.sqrt(freq)


def sharpe(series, rf=0.0, freq=252):
    s = float(series.std())
    if s == 0: return 0.0
    return (float(series.mean()) - rf/freq) / s * np.sqrt(freq)


def hist_var_es(series, alpha=0.95):
    v = series.dropna().values
    if len(v) == 0: return 0.0, 0.0
    v_sorted = np.sort(v)
    k = int((1.0 - alpha) * len(v_sorted))
    k = max(0, min(len(v_sorted)-1, k))
    VaR = float(v_sorted[k])
    ES  = float(v_sorted[:k+1].mean()) if k >= 0 else VaR
    return VaR, ES


def max_dd(equity):
    if len(equity) == 0: return 0.0
    return float((equity/equity.cummax() - 1.0).min())


def try_min_vol_weights(px_hist, min_w=0.0, max_w=0.35):
    """
    Versuche Min-Vol mit PyPortfolioOpt.
    Rückgabe: (weights: pd.Series, used_optimizer: bool)
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
            w = w / w.sum()
        return w, True
    except Exception:
        # Fallback: Equal-Weight
        cols = list(px_hist.columns)
        if len(cols) == 0:
            return pd.Series(dtype=float), False
        w = pd.Series(1.0/len(cols), index=cols, dtype=float)
        return w, False


# ---------------- Sidebar (Inputs) ----------------
st.sidebar.header("Settings")

preset = st.sidebar.checkbox("Use KAYEN preset universe", value=True)
default_tickers = ["EURUSD=X", "SPY", "TLT", "GLD", "BTC-USD", "ETH-USD"] if preset else []
tickers = st.sidebar.text_input("Tickers (comma separated)", ", ".join(default_tickers))
tickers = [t.strip() for t in tickers.split(",") if t.strip()]

years = st.sidebar.slider("Years of history", 2, 10, 5)
min_w = st.sidebar.slider("Min weight per asset", 0.00, 0.20, 0.00, 0.01)
max_w = st.sidebar.slider("Max weight per asset", 0.10, 0.70, 0.35, 0.01)
crypto_cap = st.sidebar.slider("Crypto cap (total)", 0.00, 0.60, 0.20, 0.01)
alpha = st.sidebar.slider("Risk alpha (VaR/CVaR)", 0.80, 0.99, 0.95, 0.01)
portfolio_value = st.sidebar.number_input("Portfolio value (EUR)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")

if len(tickers) == 0:
    st.warning("Bitte Ticker eingeben.")
    st.stop()

# ---------------- Data ----------------
try:
    px = load_prices(tickers, years=years)
    if px.empty:
        st.error("Keine Preisdaten geladen—prüfe Ticker.")
        st.stop()
    rets = to_returns(px, log=True).dropna()
except Exception as e:
    st.error(f"Datenfehler: {e}")
    st.stop()

# ---------------- Optimize (target weights) ----------------
weights, used_opt = try_min_vol_weights(px, min_w=min_w, max_w=max_w)

# Crypto-Kappung (global target)
if crypto_cap > 0.0 and len(weights):
    crypto_names = ("BTC","ETH","SOL","DOGE","ADA")
    ca = [t for t in weights.index if any(sym in t for sym in crypto_names)]
    cs = float(weights.reindex(ca).fillna(0.0).sum()) if ca else 0.0
    if cs > crypto_cap:
        scale = crypto_cap / cs
        weights.loc[ca] = weights.loc[ca] * scale
        non_ca = [t for t in weights.index if t not in ca]
        rem = 1.0 - float(weights.sum())
        if non_ca and rem > 0:
            denom = float(weights.loc[non_ca].sum())
            if denom <= 0:
                weights.loc[non_ca] += rem / len(non_ca)
            else:
                weights.loc[non_ca] += rem * (weights.loc[non_ca] / denom)

# Portfolio Stats (current target)
port = rets.dot(weights.values)
cum  = (1 + port).cumprod()
ann_ret, ann_vol = ann_stats(port)
sr = sharpe(port)
VaR, ES = hist_var_es(port, alpha=alpha)
MDD = max_dd(cum)

# Rolling
roll_sharpe = port.rolling(60).apply(lambda x: (x.mean()/x.std())*np.sqrt(252) if np.std(x)!=0 else 0.0, raw=True)
roll_cum = (1+port).cumprod()
roll_dd = (roll_cum/roll_cum.cummax() - 1.0)


# ---------------- Tabs ----------------
tab_overview, tab_risk, tab_backtest, tab_trades = st.tabs(["Overview", "Risk", "Backtest", "Trades"])


# ---------------- Overview ----------------
def kpi_card(label:str, value:str, delta:str|None=None):
    delta_html = f"<span style='color:#10b981; font-weight:600; font-size:.95rem; margin-left:.35rem'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-big">{value}{delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab_overview:
    a,b,c,d = st.columns(4)
    with a: kpi_card("Annualized Return", f"{ann_ret:.2%}")
    with b: kpi_card("Annualized Volatility", f"{ann_vol:.2%}")
    with c: kpi_card("Sharpe", f"{sr:.2f}")
    with d: kpi_card("Max Drawdown", f"{MDD:.2%}")
    st.caption(f"Optimizer: {'PyPortfolioOpt (min vol)' if used_opt else 'Equal-Weight Fallback'}")

    # Tables
    L, R = st.columns([1.25, 1.0])
    with L:
        st.subheader("Portfolio Overview (target)")
        ytd = {t: rets[t].loc[rets.index.year == rets.index[-1].year].sum() for t in px.columns}
        df_over = pd.DataFrame({
            "Weight": weights.round(4),
            "YTD Return": pd.Series(ytd).round(4),
        }).sort_index()
        st.dataframe(df_over, use_container_width=True)

    with R:
        st.subheader("Risk Metrics")
        df_risk = pd.DataFrame({
            "Ann. Return": [ann_ret],
            "Ann. Vol": [ann_vol],
            "Sharpe": [sr],
            f"VaR({int(alpha*100)}%)": [VaR],
            f"CVaR({int(alpha*100)}%) (approx)": [ES],
            "MaxDD": [MDD],
        }).round(4)
        st.dataframe(df_risk, use_container_width=True)


# ---------------- Risk ----------------
with tab_risk:
    st.subheader("Correlation Heatmap")
    corr = rets.corr().round(2)
    hm = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale=[[0.00, "#05161c"], [0.25, "#0b3a46"], [0.50, "#0f766e"], [0.75, "#14b8a6"], [1.00, "#99f6e4"]],
            zmin=-1, zmax=1, colorbar=dict(title="ρ", outlinewidth=0)
        )
    )
    hm.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=460)
    st.plotly_chart(hm, use_container_width=True)

    st.markdown("### Rolling Sharpe (60d)")
    fig_rs = go.Figure()
    fig_rs.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="Rolling Sharpe (60d)", mode="lines"))
    fig_rs.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=260)
    st.plotly_chart(fig_rs, use_container_width=True)

    st.markdown("### Drawdown")
    dd = (roll_cum/roll_cum.cummax()-1.0)
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", mode="lines"))
    fig_dd.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=260)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Alerts
    alerts = []
    crypto_names = ("BTC","ETH","SOL","DOGE","ADA")
    c_assets = [t for t in weights.index if any(sym in t for sym in crypto_names)]
    c_sum = float(weights.reindex(c_assets).fillna(0.0).sum()) if c_assets else 0.0
    if c_sum > crypto_cap: alerts.append(f"Crypto exposure {c_sum:.1%} exceeds cap {crypto_cap:.0%}.")
    if not roll_sharpe.dropna().empty and roll_sharpe.iloc[-1] < 1.0: alerts.append("Sharpe < 1.0 (60d).")
    if not dd.dropna().empty and dd.iloc[-1] < -0.15: alerts.append("Drawdown breached -15%.")

    st.markdown("<hr/>", unsafe_allow_html=True)
    if alerts:
        for a in alerts: st.warning(a)
    else:
        st.success("✅ No alerts.")


# ---------------- Backtest (Monthly Rebalance) ----------------
@st.cache_data(show_spinner=False, ttl=300)
def load_prices_bt(tickers, years):
    return load_prices(tickers, years=years)

def rebalance_backtest(px, rets, dates, min_w, max_w, crypto_cap, cost_bps):
    """
    px: Preise DF | rets: Returns DF | dates: Rebalance-Daten (DatetimeIndex)
    """
    w = None
    last_w = None
    eq = []
    equity = 1.0

    for dt in rets.index:
        if dt in dates:
            px_hist = px.loc[:dt].dropna()
            if px_hist.empty or px_hist.shape[1] == 0:
                w_new = pd.Series(1.0/len(px.columns), index=px.columns, dtype=float)
            else:
                tmp = try_min_vol_weights(px_hist, min_w=min_w, max_w=max_w)
                # robustes Unpacking (kann (weights, used) liefern)
                w_new = tmp[0] if isinstance(tmp, (tuple, list)) else tmp

            # w_new robust in Series + Ausrichtung
            if not isinstance(w_new, pd.Series):
                arr = np.asarray(w_new).ravel() if w_new is not None else np.array([])
                base = pd.Series(0.0, index=px_hist.columns, dtype=float)
                take = min(arr.size, len(base))
                if take > 0: base.iloc[:take] = arr[:take]
                w_new = base
            else:
                w_new = w_new.reindex(px_hist.columns).fillna(0.0).astype(float)

            # Bounds + Normalisierung
            w_new = w_new.clip(lower=min_w, upper=max_w)
            s = float(w_new.sum())
            if s <= 0:
                w_new = pd.Series(1.0/len(w_new), index=w_new.index, dtype=float)
            else:
                w_new /= s

            # Crypto-Cap
            if crypto_cap > 0.0:
                crypto_names = ("BTC","ETH","SOL","DOGE","ADA")
                ca = [t for t in w_new.index if any(sym in t for sym in crypto_names)]
                cs = float(w_new.reindex(ca).fillna(0.0).sum()) if ca else 0.0
                if cs > crypto_cap and cs > 0:
                    scale = crypto_cap / cs
                    w_new.loc[ca] = w_new.loc[ca] * scale
                    nc = [t for t in w_new.index if t not in ca]
                    rem = 1.0 - float(w_new.sum())
                    if nc and rem > 0:
                        denom = float(w_new.loc[nc].sum())
                        if denom <= 0:
                            w_new.loc[nc] += rem / len(nc)
                        else:
                            w_new.loc[nc] += rem * (w_new.loc[nc] / denom)

            # Kosten auf Turnover
            if last_w is not None:
                turnover = float((w_new - last_w).abs().sum())
                equity *= (1.0 - (cost_bps/10000.0) * turnover)

            w = w_new.copy()
            last_w = w.copy()

        if w is None:
            w = pd.Series(1.0/len(rets.columns), index=rets.columns, dtype=float)

        r = float(rets.loc[dt].reindex(w.index).fillna(0.0).dot(w))
        equity *= (1.0 + r)
        eq.append(equity)

    return pd.Series(eq, index=rets.index, name="Portfolio")


with tab_backtest:
    st.subheader("Backtest · Performance vs Benchmark")

    bt_years = st.slider("Backtest years", 2, 10, years, key="bt_years")
    cost_bps = st.number_input("Transaction cost (bps per turnover)", 0, 200, 10, step=5, key="bt_cost_bps")

    px_bt = load_prices_bt(tickers, years=bt_years)
    rets_bt = to_returns(px_bt, log=True).dropna()

    # Rebalance: erster Handelstag jedes Monats
    rebal_dates = rets_bt.groupby([rets_bt.index.year, rets_bt.index.month]).head(1).index

    # Run backtest
    eq = rebalance_backtest(
        px=px_bt, rets=rets_bt, dates=rebal_dates,
        min_w=min_w, max_w=max_w, crypto_cap=crypto_cap, cost_bps=cost_bps,
    )

    # Kennzahlen (Backtest)
    bt_ret = eq.pct_change().dropna()
    n = len(bt_ret) if len(bt_ret) else 1
    bt_cagr = (eq.iloc[-1] ** (252/n)) - 1 if len(eq) else 0.0
    bt_vol  = bt_ret.std() * np.sqrt(252) if len(bt_ret) else 0.0
    bt_sha  = (bt_ret.mean()/bt_ret.std()) * np.sqrt(252) if len(bt_ret) and bt_ret.std()!=0 else 0.0
    bt_mdd  = (eq/eq.cummax() - 1.0).min() if len(eq) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Backtest CAGR", f"{bt_cagr:.2%}")
    m2.metric("Backtest Vol", f"{bt_vol:.2%}")
    m3.metric("Backtest Sharpe", f"{bt_sha:.2f}")
    m4.metric("Backtest MaxDD", f"{bt_mdd:.2%}")

    # Benchmark
    bench_ticker = st.selectbox("Benchmark", ["SPY","ACWI","QQQ","IEF","GLD"], index=0, key="bench_sel")

    @st.cache_data(show_spinner=False, ttl=300)
    def load_bench(ticker, years):
        return yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)["Close"].dropna()

    bench_px = load_bench(bench_ticker, bt_years)

    common = eq.index.intersection(bench_px.index)
    if len(common) == 0:
        st.warning("Keine gemeinsame Historie mit Benchmark.")
    else:
        bench_eq = bench_px.loc[common] / bench_px.loc[common].iloc[0]
        port_eq  = eq.loc[common] / eq.loc[common].iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=common, y=port_eq.values,  name="Portfolio",  mode="lines"))
        fig.add_trace(go.Scatter(x=common, y=bench_eq.values, name=bench_ticker, mode="lines"))
        fig.update_layout(title="Portfolio vs. Benchmark (Index=1.0)", xaxis_title="Date", yaxis_title="Index",
                          margin=dict(l=10, r=10, t=40, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)

        outp = (port_eq/bench_eq) - 1.0
        st.write(f"Outperformance vs {bench_ticker}: {float(outp.iloc[-1]):.2%}")


# ---------------- Trades / Rebalance Planner ----------------
with tab_trades:
    st.subheader("Rebalance Planner")

    if "current_weights" not in st.session_state:
        st.session_state.current_weights = weights.copy()

    cur_df = pd.DataFrame({"Current": st.session_state.current_weights.round(4)})
    edited = st.data_editor(cur_df, use_container_width=True, key="cur_edit")
    current = edited["Current"].reindex(weights.index).fillna(0.0)

    st.caption(f"Current sum: {current.sum():.3f}  ·  Target sum: {weights.sum():.3f}")
    if abs(current.sum() - 1.0) > 0.02:
        st.warning("Gewichtssumme ist nicht ~1.00 – vor Trade planen normalisieren.")

    delta = (weights - current).rename("Delta (target-current)")
    last_prices = px.iloc[-1].reindex(weights.index).fillna(0.0)
    notional = (delta * float(portfolio_value)).rename("Trade Notional (EUR)")

    plan = pd.concat([current.rename("Current"), weights.rename("Target"),
                      delta, last_prices.rename("Last Price"), notional], axis=1).round(6)

    st.dataframe(plan, use_container_width=True)
    buf = io.BytesIO(); plan.to_csv(buf, index=True)
    st.download_button("Download Rebalance Plan (CSV)", data=buf.getvalue(),
                       file_name="rebalance_plan.csv", mime="text/csv")
