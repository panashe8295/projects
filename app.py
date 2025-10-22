# app.py
import numpy as np
import pandas as pd
import altair as alt
import pydeck as pdk
import requests  # for GeoJSON fetch

import matplotlib
matplotlib.use("Agg")  # headless backend (for Streamlit Cloud, servers, etc.)
import matplotlib.pyplot as plt

import streamlit as st
import yfinance as yf
from datetime import datetime


st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# ------------------ Config ------------------
DEFAULT_TICKERS = ["SPY", "AMZN", "BABA", "AAPL", "MSFT", "NVDA"]
DEFAULT_WEIGHTS = {"SPY": 0.10, "AMZN": -0.50, "BABA": 0.40}  # others default to 0.0
COMMON_TICKERS = sorted(list(set(DEFAULT_TICKERS + """
AAPL MSFT NVDA META GOOGL AMZN TSLA JPM BAC WFC V MA KO PEP DIS INTC AMD NFLX XOM CVX TMO LLY UNH JNJ PFE ORCL CRM COST HD LOW
BRK.B SPY QQQ IWM EFA EEM TLT GLD BABA BIDU TSM SAP SONY ASML NKE SBUX RTX GE BA
""".split())))

# ------------------ Helpers (yfinance only) ------------------
@st.cache_data(show_spinner=False, ttl=60*20)
def fetch_history(tickers, period):
    """Fetch Adj Close via yfinance for all tickers."""
    data = yf.Tickers(" ".join(tickers)).history(period=period, auto_adjust=False)
    if data is None or data.empty:
        return pd.DataFrame()
    close_like = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
    df = data[close_like].copy()
    # Flatten multiindex columns to ticker symbols
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    return df

def compute_indicators(prices, ma1=20, ma2=50, rsiw=14):
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

def random_optimize(returns, objective, n_sims=20000, allow_short=True, seed=42):
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
        if pv == 0:
            continue
        val = pr/pv if objective == "Sharpe" else (pr if objective == "Return" else pv)
        if (objective in ("Sharpe","Return") and val > best_val) or (objective == "Volatility" and val < best_val):
            best_val, best_w = val, w
    if best_w is None:
        best_w = np.ones(n) / n
    return pd.Series(best_w, index=mu.index)

@st.cache_data(show_spinner=False, ttl=60*30)
def load_fundamentals(tickers):
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
st.title("Portfolio Dashboard")

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

# ------------------ Load prices from yfinance ------------------
prices = fetch_history(tickers, period)
if prices.empty:
    st.error("No data returned by yfinance for the selected tickers/period.")
    st.stop()

prices = prices.dropna(how="all").ffill().dropna().sort_index()
# Keep only requested tickers that actually returned data
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

# ------------------ Portfolio series & stats (robust) ------------------
# Recompute from final weights
port_ret_series = returns[weights.index] @ weights
port_curve = (1.0 + port_ret_series).cumprod()
port_curve.name = "Portfolio"

# Daily stats
port_mu_daily = float(port_ret_series.mean())
port_sigma_daily = float(port_ret_series.std())
port_sharpe_daily = (port_mu_daily / port_sigma_daily) if port_sigma_daily != 0 else None
port_cumret = float(port_curve.iloc[-1] - 1.0)
w_sum = float(weights.sum())

# Optional annualization
st.subheader("Final statistics")
annualize = st.checkbox("Show annualized figures (252 trading days)", value=False)
if annualize:
    ann_factor = np.sqrt(252)
    port_mu = port_mu_daily * 252.0
    port_sigma = port_sigma_daily * ann_factor
    port_sharpe = (port_mu / port_sigma) if port_sigma != 0 else None
    mean_label = "Mean (annualized)"
    vol_label = "Volatility (annualized)"
    sharpe_label = "Sharpe (annualized)"
else:
    port_mu = port_mu_daily
    port_sigma = port_sigma_daily
    port_sharpe = port_sharpe_daily
    mean_label = "Mean (daily μ)"
    vol_label = "Volatility (daily σ)"
    sharpe_label = "Sharpe (μ/σ)"

# Display stats in same font (table)
port_df = pd.DataFrame([{
    "Cumulative Return": port_cumret,
    mean_label: port_mu,
    vol_label: port_sigma,
    sharpe_label: (None if port_sharpe is None or not np.isfinite(port_sharpe) else float(port_sharpe)),
    "Sum of Weights": w_sum,
}])
st.dataframe(
    port_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Cumulative Return": st.column_config.NumberColumn(format="%.2f%%"),
        mean_label:         st.column_config.NumberColumn(format="%.6f"),
        vol_label:          st.column_config.NumberColumn(format="%.6f"),
        sharpe_label:       st.column_config.NumberColumn(format="%.2f"),
        "Sum of Weights":   st.column_config.NumberColumn(format="%.4f"),
    },
)

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

# ------------------ Preload fundamentals once ------------------
fundamentals_all = load_fundamentals(list(prices.columns))

# ------------------ Tabs ------------------
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
    mu2, vol2, cov2 = returns.mean(), returns.std(), returns.cov()
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
        r = float(mu2.values @ w)
        v = float(np.sqrt(w @ cov2.values @ w))
        pts.append({"Vol": v, "Ret": r})
    df_cloud = pd.DataFrame(pts)
    df_assets = pd.DataFrame({"Ticker": cols, "Vol": vol2.values, "Ret": mu2.values})
    df_port = pd.DataFrame([{"Item":"Portfolio","Vol": float(port_sigma_daily), "Ret": float(port_mu_daily)}])

    ch_cloud = alt.Chart(df_cloud).mark_point(opacity=0.18).encode(x="Vol:Q", y="Ret:Q")
    ch_assets = alt.Chart(df_assets).mark_point(size=80, color="black").encode(
        x="Vol:Q", y="Ret:Q", tooltip=["Ticker","Vol","Ret"], shape=alt.value("cross"))
    ch_port = alt.Chart(df_port).mark_point(size=220, color="red").encode(x="Vol:Q", y="Ret:Q")
    st.altair_chart((ch_cloud + ch_assets + ch_port).properties(height=380), use_container_width=True)

with tab3:
    st.subheader("Fundamentals (best-effort)")
    st.dataframe(fundamentals_all, use_container_width=True)

    st.subheader("AI-style commentary")
    w_long = weights[weights > 0].sum()
    w_short = -weights[weights < 0].sum()
    skew = "balanced" if abs(w_long - w_short) < 0.2 else ("long-biased" if w_long > w_short else "short-biased")
    tilt = "defensive" if w_short > 0.3 else "growth-tilted" if w_long > 0.7 else "mixed"
    st.write(
        f"Your allocation looks **{skew}** and **{tilt}**. On this dataset, daily μ = **{port_mu:.4f}**, "
        f"σ = **{port_sigma:.4f}**, Sharpe = **{(port_sharpe if (port_sharpe is not None and np.isfinite(port_sharpe)) else float('nan')):.2f}**. "
        f"Consider adding low-correlation assets to stabilize variance and watching concentration risk."
    )

with tab4:
    st.subheader("World footprint")

    # --- Fundamentals → issuer country (with overrides for some tickers) ---
    df_f = fundamentals_all[["Ticker", "Country"]].copy()

    ISSUER_COUNTRY_OVERRIDE = {
        "BABA": "China",
        "AMZN": "United States",
        "AAPL": "United States",
        "MSFT": "United States",
        "NVDA": "United States",
        "SPY": "United States",  # ETF simplification
        "TSM": "Taiwan",
        "BIDU": "China",
        "TSLA": "United States",
        "META": "United States",
        "GOOGL": "United States",
    }
    df_f["Country"] = df_f.apply(
        lambda r: ISSUER_COUNTRY_OVERRIDE.get(str(r["Ticker"]).upper(), r["Country"]),
        axis=1
    )
    df_f = df_f.dropna(subset=["Country"]).copy()

    # Presence by country (one color per country if any ticker from that country)
    country_tickers = (
        df_f.groupby("Country")["Ticker"]
        .apply(lambda s: sorted(set(s.astype(str))))
        .to_dict()
    )
    active_countries = list(country_tickers.keys())

    if not active_countries:
        st.info("No country data available from yfinance fundamentals.")
    else:
        # Map fundamentals names -> GeoJSON names
        COUNTRY_NAME_FIX = {
            "United States": "United States of America",
            "South Korea": "Republic of Korea",
            "Russia": "Russian Federation",
            "Iran": "Iran (Islamic Republic of)",
            "Syria": "Syrian Arab Republic",
            "Viet Nam": "Vietnam",
            "Czech Republic": "Czechia",
            "Ivory Coast": "Côte d'Ivoire",
            "Congo (Kinshasa)": "Democratic Republic of the Congo",
            "Congo (Brazzaville)": "Congo",
            "Venezuela": "Venezuela (Bolivarian Republic of)",
            "Tanzania": "United Republic of Tanzania",
            "Laos": "Lao People's Democratic Republic",
            "Bolivia": "Bolivia (Plurinational State of)",
            "Brunei": "Brunei Darussalam",
            "South Sudan": "South Sudan",
            "Hong Kong": "Hong Kong",
        }
        rev_fix = {}
        for f_name, g_name in COUNTRY_NAME_FIX.items():
            rev_fix.setdefault(g_name, []).append(f_name)

        # Load world GeoJSON polygons
        url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        try:
            gj = requests.get(url, timeout=20).json()
        except Exception:
            gj = None
            st.error("Could not load world polygons for the map.")

        if gj:
            # Color palette & mapping (one color per active country)
            palette_rgb = [
                (220,  20,  60),   # red
                ( 30, 144, 255),   # blue
                (255, 140,   0),   # orange
                ( 34, 139,  34),   # green
                (138,  43, 226),   # purple
                (  0, 206, 209),   # teal
                (255, 105, 180),   # pink
                (205, 133,  63),   # brown
                (255, 215,   0),   # gold
                ( 70, 130, 180),   # steel
            ]
            color_map = {c: palette_rgb[i % len(palette_rgb)] for i, c in enumerate(active_countries)}

            # Enrich GeoJSON features with fill/line color and tickers
            for f in gj.get("features", []):
                name = f.get("properties", {}).get("name", "")
                fundamentals_name = None

                if name in active_countries:
                    fundamentals_name = name
                elif name in rev_fix:
                    for cand in rev_fix[name]:
                        if cand in active_countries:
                            fundamentals_name = cand
                            break

                if fundamentals_name:
                    r, g, b = color_map[fundamentals_name]
                    fill = [r, g, b, 220]             # colored country (opaque-ish)
                    line = [255, 255, 255, 200]       # bright border
                    tickers_here = ", ".join(country_tickers[fundamentals_name])
                else:
                    # Darker grey for non-active countries
                    fill = [20, 20, 20, 230]
                    line = [60, 60, 60, 200]
                    tickers_here = ""

                props = f.setdefault("properties", {})
                props["fill_color"] = fill
                props["line_color"] = line
                props["tickers"] = tickers_here

            layer = pdk.Layer(
                "GeoJsonLayer",
                gj,
                stroked=True,
                filled=True,
                get_fill_color="properties.fill_color",
                get_line_color="properties.line_color",
                get_line_width=1.2,
                pickable=True,
            )

            view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2)
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v11",  # darker basemap
                tooltip={"text": "{name}\nTickers: {tickers}"}
            )
            st.pydeck_chart(deck)

            # Legend: country → color
            st.markdown("**Map legend (country → color)**")
            legend_bits = []
            for c in active_countries:
                r, g, b = color_map[c]
                bit = f"""
                <span style="display:inline-block;margin:4px 12px 6px 0;">
                  <span style="display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:2px;margin-right:6px;"></span>
                  {c}
                </span>
                """
                legend_bits.append(bit)
            st.markdown("".join(legend_bits), unsafe_allow_html=True)

            with st.expander("Show country → tickers"):
                show_rows = []
                for c in sorted(active_countries):
                    for t in country_tickers[c]:
                        show_rows.append({"Country": c, "Ticker": t})
                st.dataframe(pd.DataFrame(show_rows), use_container_width=True)

# --------------- Last 10 days of normalized data ---------------
st.subheader("Last 10 trading days — Normalized prices (start = 1.0)")
last10 = norm.tail(10).copy()
last10.index = last10.index.strftime("%Y-%m-%d")
st.dataframe(last10.round(4), use_container_width=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Data via yfinance • Weights persist during this session")
