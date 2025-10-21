# app.py
import numpy as np
import pandas as pd
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Portfolio Lab — Table + Optimizer", layout="wide")

# ------------------ Config ------------------
DATA_DIR = Path("Data")
DEFAULT_TICKERS = ["SPY", "AMZN", "BABA", "AAPL", "MSFT", "NVDA"]
DEFAULT_WEIGHTS = {"SPY": 0.10, "AMZN": -0.50, "BABA": 0.40}  # others default to 0.0
COMMON_TICKERS = sorted(list(set(DEFAULT_TICKERS + """
AAPL MSFT NVDA META GOOGL AMZN TSLA JPM BAC WFC V MA KO PEP DIS INTC AMD NFLX XOM CVX TMO LLY UNH JNJ PFE ORCL CRM COST HD LOW
BRK.B SPY QQQ IWM EFA EEM TLT GLD BABA BIDU TSM SAP SONY ASML NKE SBUX RTX GE BA
""".split())))

# ------------------ Helpers ------------------
def load_adj_close_csv(fp: Path, label: str) -> pd.Series:
    """Load Adj Close from a yfinance-style CSV with multiindex columns."""
    df = pd.read_csv(fp, header=[0, 1], index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(1):
            s = df.xs("Adj Close", axis=1, level=1)
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1: s = s.iloc[:, 0]
        elif "Adj Close" in df.columns.get_level_values(0):
            s = df.xs("Adj Close", axis=1, level=0)
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1: s = s.iloc[:, 0]
        else:
            raise KeyError("'Adj Close' not found in multiindex columns")
    else:
        s = df["Adj Close"]
    s.name = label
    return s

def try_load_ticker_local(ticker: str) -> pd.Series | None:
    fp = DATA_DIR / f"{ticker}.csv"
    if not fp.exists(): return None
    try:
        return load_adj_close_csv(fp, ticker)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=60*20)
def fetch_history(tickers: list[str], period: str) -> pd.DataFrame:
    """Fallback: fetch Adj Close via yfinance for any missing tickers."""
    df = yf.Tickers(" ".join(tickers)).history(period=period, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    close_like = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
    data = df[close_like].copy()
    data.columns = [c[1] if isinstance(c, tuple) else c for c in data.columns]  # flatten col names
    return data

def compute_indicators(prices: pd.Series, ma1=20, ma2=50, rsiw=14):
    s = prices.dropna()
    sma1 = s.rolling(ma1).mean()
    sma2 = s.rolling(ma2).mean()
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(rsiw).mean()
    avg_loss = loss.rolling(rsiw).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return sma1, sma2, rsi

def random_optimize(returns: pd.DataFrame, objective: str, n_sims=20000, allow_short=True, seed=42):
    rng = np.random.default_rng(seed)
    mu, cov = returns.mean(), returns.cov()
    n = len(mu)
    best_w = None
    best_val = -np.inf if objective in ("Sharpe","Return") else np.inf
    for _ in range(n_sims):
        if allow_short:
            w = rng.normal(0, 1, size=n); s = w.sum(); w = (w/s) if s != 0 else w
        else:
            w = rng.random(n); w = w / w.sum()
        pr = float(mu.values @ w)
        pv = float(np.sqrt(w @ cov.values @ w))
        if pv == 0: continue
        val = pr/pv if objective=="Sharpe" else (pr if objective=="Return" else pv)
        if (objective in ("Sharpe","Return") and val > best_val) or (objective=="Volatility" and val < best_val):
            best_val, best_w = val, w
    if best_w is None: best_w = np.ones(n)/n
    return pd.Series(best_w, index=mu.index)

@st.cache_data(show_spinner=False, ttl=60*30)
def load_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}
        rows.append({
            "Ticker": t,
            "Short Name": info.get("shortName"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Country": info.get("country"),
            "Trailing PE": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "Dividend Yield": info.get("dividendYield"),
        })
    return pd.DataFrame(rows)

# ------------------ UI: Header & controls ------------------
st.title("Portfolio Lab — Table + Optimizer")
st.caption("Reads local CSVs from `Data/` when available, otherwise fetches via yfinance.")

left, right = st.columns([1, 3])
with left:
    period = st.selectbox("History window", ["6mo", "1y", "3y", "5y", "10y"], index=1)
    tickers = st.multiselect(
        "Assets",
        options=COMMON_TICKERS,
        default=DEFAULT_TICKERS,
        placeholder="Type to search; you can add new tickers",
        accept_new_options=True,
    )

# Ensure uppercase & unique
tickers = sorted(list(dict.fromkeys([t.upper().strip() for t in tickers if t.strip()])))

if not tickers:
    st.info("Pick at least one ticker to continue.")
    st.stop()

# ------------------ Load prices: local first, then fetch missing ------------------
series = []
missing = []
for t in tickers:
    s = try_load_ticker_local(t)
    if s is not None:
        series.append(s)
    else:
        missing.append(t)

if missing:
    fetched = fetch_history(missing, period)
    for t in missing:
        if t in fetched.columns:
            series.append(fetched[t].rename(t))

if not series:
    st.error("No data found (local CSVs missing and yfinance returned empty).")
    st.stop()

prices = pd.concat(series, axis=1).dropna(how="all").ffill().dropna().sort_index()
# Align to selected tickers (drop any with empty data)
prices = prices[[c for c in tickers if c in prices.columns]]
if prices.empty:
    st.error("No overlapping data across selected tickers.")
    st.stop()

returns = prices.pct_change().dropna()

# ------------------ Build the editable weights table ------------------
norm = prices / prices.iloc[0]
cumret = norm.iloc[-1] - 1.0
mu = returns.mean()
sigma = returns.std()
asset_sharpe = mu / sigma

# Initialize/persist weights
if "weights" not in st.session_state:
    st.session_state.weights = {t: float(DEFAULT_WEIGHTS.get(t, 0.0)) for t in prices.columns}
# add missing keys / prune removed
for t in prices.columns:
    st.session_state.weights.setdefault(t, 0.0)
for t in list(st.session_state.weights.keys()):
    if t not in prices.columns:
        st.session_state.weights.pop(t, None)

table_df = pd.DataFrame({
    "Ticker": prices.columns,
    "Cumulative Return": [float(cumret[t]) for t in prices.columns],
    "Sharpe (asset)": [float(asset_sharpe[t]) if np.isfinite(asset_sharpe[t]) else np.nan for t in prices.columns],
    "Weight": [float(st.session_state.weights.get(t, 0.0)) for t in prices.columns],
})

st.subheader("Assets Table (edit weights in the last column)")
edited_df = st.data_editor(
    table_df,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Cumulative Return": st.column_config.NumberColumn(format="%.2f%%"),
        "Sharpe (asset)": st.column_config.NumberColumn(format="%.2f"),
        "Weight": st.column_config.NumberColumn(format="%.4f", step=0.01),
    },
    disabled=["Ticker", "Cumulative Return", "Sharpe (asset)"],
    use_container_width=True,
)

# Pull weights back from table (keep order)
weights = pd.Series({row["Ticker"]: float(row["Weight"]) for _, row in edited_df.iterrows()})
weights = weights.reindex(prices.columns).fillna(0.0)

# Buttons & options for weights
colA, colB, colC, colD = st.columns([1,1,2,2])
with colA:
    if st.button("Equal allocation"):
        eq = 1.0 / len(weights) if len(weights) else 0.0
        weights = pd.Series(eq, index=weights.index)
with colB:
    force_sum = st.checkbox("Force sum(weights)=1.0", value=False)
with colC:
    optimizer_obj = st.selectbox("Optimize objective", ["Sharpe", "Return", "Volatility"])
with colD:
    sims = st.slider("Sims", 2000, 100000, 20000, 2000)
    allow_short = st.checkbox("Allow shorting (optimizer)", value=True)

if st.button("Optimize"):
    w_opt = random_optimize(returns[weights.index], optimizer_obj, n_sims=sims, allow_short=allow_short, seed=42)
    weights = w_opt

if force_sum and weights.sum() != 0:
    weights = weights / weights.sum()

# Persist new weights into session state so they survive reruns
st.session_state.weights.update({t: float(weights[t]) for t in weights.index})

# ------------------ Portfolio series & stats ------------------
port_ret_series = returns @ weights
port_curve = (1.0 + port_ret_series).cumprod()
port_curve.name = "Portfolio"
port_mu = float(port_ret_series.mean())
port_sigma = float(port_ret_series.std())
port_sharpe = (port_mu / port_sigma) if port_sigma != 0 else np.nan
port_cumret = float(port_curve.iloc[-1] - 1.0)
w_sum = float(weights.sum())

# ------------------ Main chart: normalized + portfolio ------------------
st.subheader("Performance (normalized to 1.0)")
fig = plt.figure(figsize=(10, 5))
ax = fig.gca()
(norm.plot(ax=ax, lw=1.6, alpha=0.9))
port_curve.plot(ax=ax, lw=4.0, color="red", label="Portfolio")
ax.set_title("Assets & Portfolio (Normalized)")
ax.set_ylabel("Growth of $1")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# ------------------ Portfolio final stats (table: same font) ------------------
st.subheader("Portfolio — Final Stats")
port_df = pd.DataFrame([{
    "Cumulative Return": port_cumret,
    "Mean (daily μ)": port_mu,
    "Volatility (daily σ)": port_sigma,
    "Sharpe (μ/σ)": port_sharpe,
    "Sum of Weights": w_sum,
}])
st.dataframe(
    port_df.style.format({
        "Cumulative Return": "{:.2%}",
        "Mean (daily μ)": "{:.6f}",
        "Volatility (daily σ)": "{:.6f}",
        "Sharpe (μ/σ)": "{:.2f}",
        "Sum of Weights": "{:.4f}",
    }),
    hide_index=True,
    use_container_width=True
)

# ------------------ Tabs: Indicators • Scatter • Fundamentals & Commentary • Map ------------------
tab1, tab2, tab3, tab4 = st.tabs(["Indicators", "Risk/Return", "Fundamentals & Commentary", "World Footprint"])

with tab1:
    st.subheader("Technical indicators")
    t_sel = st.selectbox("Ticker", options=list(prices.columns))
    c1, c2, c3 = st.columns(3)
    ma1 = c1.number_input("SMA window 1", min_value=2, value=20, step=1)
    ma2 = c2.number_input("SMA window 2", min_value=2, value=50, step=1)
    rsiw = c3.number_input("RSI window", min_value=2, value=14, step=1)
    sma1, sma2, rsi = compute_indicators(prices[t_sel], ma1=ma1, ma2=ma2, rsiw=rsiw)

    df_price = pd.DataFrame({"Date": prices.index, "Price": prices[t_sel], f"SMA {ma1}": sma1, f"SMA {ma2}": sma2})
    st.altair_chart(
        alt.Chart(df_price.melt("Date")).mark_line().encode(
            x="Date:T", y=alt.Y("value:Q", title="Price", scale=alt.Scale(zero=False)),
            color="variable:N", tooltip=["Date:T","variable:N","value:Q"]
        ).properties(height=320),
        use_container_width=True
    )
    df_rsi = pd.DataFrame({"Date": prices.index, "RSI": rsi})
    bands = alt.Chart(pd.DataFrame({"y":[30,70]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    st.altair_chart(
        alt.Chart(df_rsi).mark_line().encode(x="Date:T", y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0,100])))
        .properties(height=180) + bands,
        use_container_width=True
    )

with tab2:
    st.subheader("Risk/Return scatter & random frontier cloud")
    mu, vol, cov = returns.mean(), returns.std(), returns.cov()
    cloudN = st.slider("Cloud size", 1000, 10000, 4000, 500, key="cloudN")
    allow_short_cloud = st.checkbox("Allow shorting (cloud)", value=True, key="cloudShort")
    rng = np.random.default_rng(7)
    pts = []
    cols = list(returns.columns)
    n = len(cols)
    for _ in range(cloudN):
        if allow_short_cloud:
            w = rng.normal(0, 1, size=n); s = w.sum(); w = (w/s) if s != 0 else w
        else:
            w = rng.random(n); w = w / w.sum()
        r = float(mu.values @ w)
        v = float(np.sqrt(w @ cov.values @ w))
        pts.append({"Vol": v, "Ret": r})
    df_cloud = pd.DataFrame(pts)
    df_assets = pd.DataFrame({"Ticker": cols, "Vol": vol.values, "Ret": mu.values})
    df_port = pd.DataFrame([{"Item":"Portfolio","Vol": port_sigma, "Ret": port_mu}])

    ch_cloud = alt.Chart(df_cloud).mark_point(opacity=0.18).encode(x="Vol:Q", y="Ret:Q")
    ch_assets = alt.Chart(df_assets).mark_point(size=80, color="black").encode(
        x="Vol:Q", y="Ret:Q", tooltip=["Ticker","Vol","Ret"], shape=alt.value("cross"))
    ch_port = alt.Chart(df_port).mark_point(size=220, color="red").encode(x="Vol:Q", y="Ret:Q")
    st.altair_chart((ch_cloud + ch_assets + ch_port).properties(height=380), use_container_width=True)

with tab3:
    st.subheader("Fundamentals (best-effort)")
    fundamentals = load_fundamentals(list(prices.columns))
    st.dataframe(fundamentals, use_container_width=True)

    st.subheader("AI-style commentary")
    w_long = weights[weights > 0].sum()
    w_short = -weights[weights < 0].sum()
    skew = "balanced" if abs(w_long - w_short) < 0.2 else ("long-biased" if w_long > w_short else "short-biased")
    risk_bar = returns.std().mean()
    risk_level = "low" if port_sigma < risk_bar/2 else "moderate" if port_sigma < risk_bar else "elevated"
    tilt = "defensive" if w_short > 0.3 else "growth-tilted" if w_long > 0.7 else "mixed"
    st.write(
        f"Your allocation looks **{skew}** and **{tilt}**. On this dataset, daily μ = **{port_mu:.4f}**, "
        f"σ = **{port_sigma:.4f}**, Sharpe = **{(port_sharpe if np.isfinite(port_sharpe) else float('nan')):.2f}**. "
        f"Consider adding low-correlation assets to stabilize variance and watching concentration risk."
    )

with tab4:
    st.subheader("World footprint (issuer country; bubble ∝ long weight)")
    df_f = fundamentals[["Ticker","Country"]].copy()
    df_f["Weight"] = df_f["Ticker"].map(weights.to_dict()).fillna(0.0)
    df_f["LongWeight"] = df_f["Weight"].clip(lower=0.0)

    # Quick country -> lat/lon mapping (coarse, for demo)
    COUNTRY_LL = {
        "United States": (39.8, -98.6), "China": (35.0, 103.8), "Taiwan": (23.7, 121.0), "Japan": (36.2, 138.3),
        "United Kingdom": (55.0, -3.4), "Germany": (51.2, 10.4), "France": (46.2, 2.2), "Canada": (56.1,-106.3),
        "South Korea": (36.5, 128.0), "India": (21.1, 78.0), "Brazil": (-14.2, -51.9), "Australia": (-25.3, 133.8),
        "Netherlands": (52.1, 5.3), "Switzerland": (46.8, 8.2), "Spain": (40.4, -3.7), "Italy": (41.9, 12.6),
        "Sweden": (60.1, 18.6), "Ireland": (53.1, -8.2), "Israel": (31.0, 35.0), "Mexico": (23.6, -102.5),
        "Singapore": (1.35, 103.8), "Hong Kong": (22.3, 114.2)
    }
    df_f["lat"] = df_f["Country"].map(lambda c: COUNTRY_LL.get(c, (0.0,0.0))[0])
    df_f["lon"] = df_f["Country"].map(lambda c: COUNTRY_LL.get(c, (0.0,0.0))[1])
    st.dataframe(df_f.fillna(""), use_container_width=True)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_f.dropna(subset=["lat","lon"]),
        get_position="[lon, lat]",
        get_radius="1000000 * LongWeight + 200000",
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                             tooltip={"text":"{Ticker}\n{Country}\nweight: {Weight}"}))

# --------------- Raw data peek ---------------
with st.expander("Show raw data (tail)"):
    st.write("**Prices:**")
    st.dataframe(prices.tail(), use_container_width=True)
    st.write("**Daily Returns:**")
    st.dataframe(returns.tail(), use_container_width=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Weights persist during this session")
