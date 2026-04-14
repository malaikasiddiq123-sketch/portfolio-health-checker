import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# --- 1. TERMINAL CONFIGURATION ---
st.set_page_config(page_title="Investment Portfolio Health Checker", layout="wide", page_icon="📈")

# Custom CSS for UI styling
st.markdown("""
    <style>
    /* Market Status Styling */
    .status-open {
        color: #00ffcc;
        font-weight: bold;
        text-transform: uppercase;
    }
    .status-closed {
        color: #ff4b4b;
        font-weight: bold;
    }
    
    /* Styling Tabs to look like Black Buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #000000;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        border: 1px solid #333;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1a1c24 !important;
        border: 1px solid #00ffcc !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def fetch_financial_intelligence(tickers, period):
    intelligence_hub = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            hist = ticker_obj.history(period=period)
            info = ticker_obj.info
            if not hist.empty:
                intelligence_hub[t] = {'data': hist, 'info': info}
        except Exception as e:
            st.sidebar.error(f"Could not fetch {t}: {e}")
    return intelligence_hub

# --- 3. SIDEBAR & CONTROL PANEL ---
st.sidebar.title("🛠️ Control Center")
st.sidebar.markdown("---")

# Currency Conversion Logic
currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
fx_rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79} # Example rates

currency_choice = st.sidebar.selectbox("Base Currency", ["USD", "EUR", "GBP"])
curr_symbol = currency_symbols[currency_choice]
curr_rate = fx_rates[currency_choice]

input_tickers = st.sidebar.text_area("Portfolio Asset Symbols", "AAPL, TSLA, MSFT, BTC-USD, GC=F, NVDA")
tickers = [t.strip().upper() for t in input_tickers.split(",")]

analysis_period = st.sidebar.selectbox("Analysis Horizon", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

st.sidebar.markdown("---")
st.sidebar.info("Model: Scikit-Learn Linear Regression\nUI: Streamlit Enterprise")

# --- 4. DATA PROCESSING ---
data_hub = fetch_financial_intelligence(tickers, analysis_period)

if data_hub:
    # --- HEADER SECTION ---
    st.title("Investment Portfolio Health Checker")
    
    now = datetime.now()
    # Market status logic (Simplified)
    is_open = 9 <= now.hour < 16
    status_text = "OPEN" if is_open else "CLOSED"
    status_class = "status-open" if is_open else "status-closed"
    
    st.markdown(f"Market Status: <span class='{status_class}'>{status_text}</span> | Feed: Real-Time", unsafe_allow_html=True)
    st.divider()

    # --- 5. TOP PERFORMANCE RIBBON ---
    metrics_cols = st.columns(len(data_hub))
    for i, t in enumerate(data_hub):
        df = data_hub[t]['data']
        # Apply conversion rate
        curr_price = df['Close'].iloc[-1] * curr_rate
        prev_close = df['Close'].iloc[-2] * curr_rate
        change = ((curr_price - prev_close) / prev_close) * 100
        metrics_cols[i].metric(t, f"{curr_symbol}{curr_price:,.2f}", f"{change:.2f}%")

    st.markdown("---")

    # --- 6. MULTI-TAB INTELLIGENCE SYSTEM ---
    tabs = st.tabs(["📉 Market Charts", "⚖️ Portfolio Health", "🔮 Future Trends", "📄 Raw Data Explorer"])

    with tabs[0]:
        c1, c2 = st.columns([3, 1])
        with c1:
            sel_asset = st.selectbox("Focus Asset", tickers)
            df_plot = data_hub[sel_asset]['data'].copy()
            # Convert values for plotting
            for col in ['Open', 'High', 'Low', 'Close']:
                df_plot[col] = df_plot[col] * curr_rate
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                               row_heights=[0.7, 0.3], subplot_titles=("Price Action & SMA", "Volume"))
            
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                                         low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1)
            
            sma50 = df_plot['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=df_plot.index, y=sma50, line=dict(color='cyan', width=1.5), name="50 SMA"), row=1, col=1)
            
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name="Volume", marker_color='gray'), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Asset Profile")
            info = data_hub[sel_asset].get('info', {})
            st.write(f"Sector: {info.get('sector', 'N/A')}")
            st.write(f"Beta: {info.get('beta', 'N/A')}")
            st.markdown(f"Summary: {info.get('longBusinessSummary', 'N/A')[:250]}...")

    with tabs[1]:
        st.subheader("Portfolio Health Analysis")
        
        # 1. Calculation: Calculate percentage change and handle empty data
        # Note: Using ffill() ensures we don't drop days due to minor data gaps
        all_rets = pd.DataFrame({t: data_hub[t]['data']['Close'].pct_change() for t in data_hub})
        all_rets = all_rets.fillna(method='ffill').dropna()
        
        # Check if we have sufficient data to perform analysis
        if not all_rets.empty:
            # Volatility (Risk) = Annualized Standard Deviation
            volatility = all_rets.std() * np.sqrt(252) * 100
            # Annualized Return = Mean Return
            returns = all_rets.mean() * 252 * 100
            
            risk_return = pd.DataFrame({'Risk': volatility, 'Return': returns})
            
            # --- RISK-RETURN SCATTER PLOT ---
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(x=risk_return['Risk'], y=risk_return['Return'],
                                          mode='markers+text', text=risk_return.index,
                                          marker=dict(size=15, color=risk_return['Return'], colorscale='RdYlGn')))
            fig_risk.update_layout(title="Risk vs Reward Analysis", xaxis_title="Risk (Annualized Volatility %)", 
                                   yaxis_title="Return (Annualized %)", template="plotly_dark")
            st.plotly_chart(fig_risk)

            # --- STABILITY SCORE (Health Meter) ---
            avg_vol = volatility.mean()
            # Calculate score: Lower volatility results in a higher stability score
            stability_score = max(0, min(100, 100 - (avg_vol * 2))) 
            
            # Define status message based on score
            if stability_score >= 70:
                status = "Stable"
            elif stability_score >= 40:
                status = "Moderate Risk"
            else:
                status = "Risky"
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=stability_score, 
                title={'text': f"Portfolio Health: {status}"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "cyan"},
                       'steps': [{'range': [0, 40], 'color': "red"}, 
                                 {'range': [40, 70], 'color': "orange"}, 
                                 {'range': [70, 100], 'color': "green"}]}))
            fig_gauge.update_layout(template="plotly_dark")
            st.plotly_chart(fig_gauge)
            
            st.write(f"*Portfolio Summary:* Your average portfolio risk (volatility) is {avg_vol:.2f}%. " 
                     f"Your portfolio is currently classified as *{status}*. "
                     "A higher score indicates a more stable and balanced investment strategy.")
        else:
            st.warning("Insufficient data available for calculation. Please try a longer 'Analysis Horizon' (e.g., 1y or 2y).")
    with tabs[2]:
        st.subheader("Predictive Analytics Engine")
        pred_data = []
        for t in data_hub:
            y = data_hub[t]['data']['Close'].values * curr_rate
            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            next_price = model.predict([[len(y)]])[0]
            pred_data.append({"Ticker": t, "Current": y[-1], "Predicted": next_price, "Confidence": model.score(X, y)})
        
        df_pred = pd.DataFrame(pred_data).set_index("Ticker")
        st.table(df_pred.style.format({"Current": f"{curr_symbol}{{:.2f}}", "Predicted": f"{curr_symbol}{{:.2f}}", "Confidence": "{:.2f}"}))

    with tabs[3]:
        st.subheader("Historical Data")
        sel_raw = st.selectbox("Select Asset", tickers, key="raw_sel")
        raw_df = data_hub[sel_raw]['data'].copy() * curr_rate
        st.dataframe(raw_df.sort_index(ascending=False), use_container_width=True)

else:
    st.warning("Enter tickers in the sidebar to start.")