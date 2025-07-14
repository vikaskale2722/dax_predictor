import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import time
import yfinance as yf
from pandas.tseries.offsets import BDay
from prediction_utils import DAXModelUtils

# =======================================================================================
#LOAD REAL DATA FROM YOUR PIPELINE
# =======================================================================================
dax_utils = DAXModelUtils(sequence_length=30)
df_new = dax_utils.load_and_prepare_data('combined_data_for_prediction.parquet')
dax_utils.load_model()
next_price, pred_date, top_headlines, explanation = dax_utils.predict_and_explain(df_new)

# =======================================================================================
#AUTO-CREATE CSV IF NEEDED
# =======================================================================================
def get_start_date(business_days):
    return (datetime.today() - BDay(business_days)).date()

def generate_dax_csv_if_needed():
    csv_filename = "dax_data.csv"
    if not os.path.exists(csv_filename):
        start_date = get_start_date(26)
        end_date = datetime.today().date()
        df = yf.download("^GDAXI", start=start_date, end=end_date)
        if df.empty:
            print("⚠️ Could not fetch data from Yahoo Finance.")
            return
        df = df[['Open', 'High', 'Low', 'Close']].reset_index()
        df.to_csv(csv_filename, index=False)
        print(f"✅ Created '{csv_filename}' from {start_date} to {end_date}.")

generate_dax_csv_if_needed()

# =======================================================================================
#STREAMLIT DASHBOARD
# =======================================================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "show_chart" not in st.session_state:
    st.session_state.show_chart = False
if "error" not in st.session_state:
    st.session_state.error = False

FORECAST_KEYWORDS = [
    "forecast", "prediction", "predict", "tomorrow",
    "next day", "next-day", "nextday", "dax",
    "price", "target", "move", "mov"
]

st.set_page_config(
    page_title="DAX AI Insights",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0A0E17; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stHeader"] {
        background-color: #0A0E17;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .widget-card {
        background-color: #1F2937;
        border: 1px solid #2D3748;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stExpander {
        background-color: #111827;
        border: 1px solid #2D3748;
        border-radius: 0.75rem;
    }
    .stExpander header {
        font-size: 1.25rem;
        color: #E5E7EB;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0A0E17;
        color: #A0AEC0;
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
        border-top: 1px solid #2D3748;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dax_data_from_csv():
    csv_filename = "dax_data.csv"
    if not os.path.exists(csv_filename):
        st.error(f"CSV file '{csv_filename}' not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_filename)
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# =======================================================================================
#PAGE ROUTING
# =======================================================================================
if st.session_state.page == "home":
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.title("🤖 DAX AI Insights")
        st.markdown("Welcome to the DAX Stock Prediction Dashboard!")
        st.markdown("""
            This tool uses advanced AI models (LSTM) to predict DAX index movements 
            and summarizes the current news sentiment from multiple sources.
        """)
        st.markdown("Click below to start analysis.")

        if st.button("🚀 Start Analysis"):
            with st.spinner("Running LSTM model and preparing data..."):
                for i in range(5):
                    st.write(f"Processing step {i+1}/5...")
                    time.sleep(1)
            st.session_state.page = "dashboard"
            st.rerun()

    with right_col:
        st.markdown("### 📝 Key Highlights")
        st.markdown("""
        <div style='font-size:1.2rem; line-height:1.8; margin-top:20px;'>
            📈 <b>Live DAX data</b> from Yahoo Finance.<br>
            📰 <b>Sentiment analysis</b> on latest headlines.<br>
            🤖 <b>LSTM predictions</b> for next-day DAX.<br>
            📊 Interactive visuals.<br>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.page == "dashboard":
    with st.container():
        c1, c2 = st.columns([1,5])
        with c1:
            st.markdown("### 🤖 DAX AI Insights")
        with c2:
            st.markdown(
                f"<p style='text-align:center;color:#A0AEC0;'>Today: {datetime.now():%B %d, %Y}</p>",
                unsafe_allow_html=True
            )

    main_col, side_col = st.columns([0.7,0.3])
    with main_col:
        user_msg = st.text_input("Enter your query...", key="chat_input")
        if st.button("Send"):
            txt = user_msg.lower()
            if any(kw in txt for kw in FORECAST_KEYWORDS):
                st.session_state.show_chart = True
                st.session_state.error = False
            else:
                st.session_state.show_chart = False
                st.session_state.error = True

    if st.session_state.error:
        with main_col:
            st.error("Can't give output for your query. Please ask about a forecast or prediction.")

    if st.session_state.show_chart:
        with main_col:
            st.markdown(f"### 📈 Next-Day DAX Forecast: `{pred_date.date()} -> {next_price:.2f}`")
            dax_df = load_dax_data_from_csv()
            if not dax_df.empty:
                dax_df = dax_df.dropna(subset=['Date']).sort_values("Date")
                last_trading_day = dax_df['Date'].max()
                st.write(f"Latest data point in dataset: `{last_trading_day:%B %d, %Y}`")

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=dax_df['Date'], open=dax_df['Open'],
                    high=dax_df['High'], low=dax_df['Low'],
                    close=dax_df['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))
                fig.add_trace(go.Scatter(
                    x=[pred_date], y=[next_price],
                    mode="markers+text",
                    marker=dict(size=12, color="orange", symbol="diamond"),
                    text=["Predicted"], textposition="top center"
                ))
                fig.update_layout(
                    xaxis_title='Date', yaxis_title='Price',
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=400
                )
                fig.update_xaxes(type='date')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available. Please check 'dax_data.csv'.")

        with main_col:
            with st.expander("📖 Explanation", expanded=True):
                st.markdown(explanation)

    with side_col:
        with st.container():
            st.markdown('<div class="widget-card">', unsafe_allow_html=True)
            st.markdown("##### 📰 Top Impactful Headlines")
            for i, row in top_headlines.iterrows():
                impact_color = "#16a34a" if row['impact'] > 0 else "#dc2626"
                st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                        <strong>{i+1}. {row['date'].date()}</strong> - 
                        <a href='{row['url']}' target='_blank' style='color:#60A5FA;text-decoration:none;'>
                            {row['headline'][:100]}...
                        </a>
                        <span style='background:{impact_color};color:white;
                                     padding:2px 6px;border-radius:4px;font-size:0.8rem;'>
                            Impact: {row['impact']:.4f}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            overall_score = top_headlines['impact'].mean()
            st.markdown(f"**Overall Mean Impact:** `{overall_score:.2f}`")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
      '<div class="footer">Data: Yahoo Finance · GDELT · OpenAI API • © 2025</div>',
      unsafe_allow_html=True
    )
