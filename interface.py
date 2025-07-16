import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import time
import yfinance as yf
from pandas.tseries.offsets import BDay
from prediction_utils import DAXModelUtils
import re

# =======================================================================================
# 🚀 LOAD REAL DATA FROM YOUR PIPELINE
# =======================================================================================
dax_utils = DAXModelUtils(sequence_length=26)
df_new = dax_utils.load_and_prepare_data("dax_final_with_sentiment_granular.parquet")
dax_utils.load_model()
next_price, pred_date, top_headlines, explanation = dax_utils.predict_and_explain(df_new)

# =======================================================================================
# 🚀 AUTO-CREATE CSV IF NEEDED (Extended to 1 month)
# =======================================================================================
def get_start_date(business_days):
    return (datetime.today() - BDay(business_days)).date()

def generate_dax_csv_if_needed():
    csv_filename = "dax_data.csv"
    if not os.path.exists(csv_filename):
        # Get 1 month of data (approximately 22 business days)
        start_date = get_start_date(30)  # Extended to 30 days to ensure we get enough data
        end_date = datetime.today().date()
        df = yf.download("^GDAXI", start=start_date, end=end_date)
        if df.empty:
            print("⚠️ Could not fetch data from Yahoo Finance.")
            return
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        df.to_csv(csv_filename, index=False)
        print(f"✅ Created '{csv_filename}' from {start_date} to {end_date}.")

def clean_headline(headline):
    """Clean headline by removing numbers, dates, and impact scores"""
    # Remove patterns like "433. 2025-07-08" at the beginning
    headline = re.sub(r'^\d+\.\s*\d{4}-\d{2}-\d{2}\s*', '', headline)
    
    # Remove "Impact: X.XXX" at the end
    headline = re.sub(r'\s*Impact:\s*[\d.-]+\s*,?\s*', '', headline)
    
    # Remove any remaining numbers at the beginning
    headline = re.sub(r'^\d+\.\s*', '', headline)
    
    # Clean up any remaining trailing characters
    headline = headline.strip()
    
    return headline

def get_unique_headlines(headlines_df, limit=10):
    """Get unique headlines without duplicates"""
    if headlines_df.empty:
        return headlines_df
    
    # Clean headlines first
    headlines_df = headlines_df.copy()
    headlines_df['clean_headline'] = headlines_df['headline'].apply(clean_headline)
    
    # Remove duplicates based on cleaned headline
    unique_headlines = headlines_df.drop_duplicates(subset=['clean_headline'])
    
    # Return top N unique headlines
    return unique_headlines.head(limit)

generate_dax_csv_if_needed()

# =======================================================================================
# 🚀 STREAMLIT DASHBOARD
# =======================================================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "show_news" not in st.session_state:
    st.session_state.show_news = False
if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

st.set_page_config(
    page_title="DAX AI Insights",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(135deg, #0A0E17 0%, #1a1a2e 100%);
        color: #E5E7EB;
    }
    
    [data-testid="stSidebar"] { display: none; }
    
    [data-testid="stHeader"] {
        background: rgba(10, 14, 23, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }
    
    .widget-card {
        background: rgba(31, 41, 55, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .widget-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.6);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E5E7EB 0%, #F9FAFB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #9CA3AF;
        margin-bottom: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .change-positive {
        color: #10B981;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .change-negative {
        color: #EF4444;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .predict-button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 16px 32px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        cursor: pointer;
        width: 100%;
        margin-top: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .predict-button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 2px solid rgba(59, 130, 246, 0.5);
        border-radius: 1.5rem;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3);
        animation: pulse-glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse-glow {
        0% { box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3); }
        100% { box-shadow: 0 25px 70px rgba(59, 130, 246, 0.5); }
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }
    
    .prediction-label {
        font-size: 1.5rem;
        color: #E5E7EB;
        margin-bottom: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .prediction-date {
        font-size: 1.2rem;
        color: #9CA3AF;
        font-weight: 500;
    }
    
    .action-button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    
    .action-button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.6);
        transform: translateY(-2px);
    }
    
    .secondary-button {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
    }
    
    .secondary-button:hover {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.6);
        transform: translateY(-2px);
    }
    
    .news-item {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.9) 0%, rgba(31, 41, 55, 0.9) 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 4px solid #3B82F6;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .news-item:hover {
        border-left-color: #8B5CF6;
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        transform: translateX(4px);
    }
    
    .news-item a {
        color: #60A5FA;
        text-decoration: none;
        font-weight: 500;
        line-height: 1.6;
        transition: color 0.3s ease;
    }
    
    .news-item a:hover {
        color: #A78BFA;
    }
    
    .news-number {
        color: #3B82F6;
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: 8px;
    }
    
    .section-title {
        color: #E5E7EB;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .back-button {
        background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(107, 114, 128, 0.4);
    }
    
    .back-button:hover {
        background: linear-gradient(135deg, #4B5563 0%, #374151 100%);
        box-shadow: 0 8px 30px rgba(107, 114, 128, 0.6);
        transform: translateY(-2px);
    }
    
    .analysis-section {
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .spinner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        margin: 2rem 0;
        border-radius: 1px;
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

def calculate_change(current, previous):
    if previous == 0:
        return 0, 0
    change = current - previous
    change_pct = (change / previous) * 100
    return change, change_pct

# =======================================================================================
# 🚀 LANDING PAGE WITH CHART AND CURRENT VALUES
# =======================================================================================
if st.session_state.page == "home":
    # Page header
    st.markdown('<h1 class="main-title">🤖 DAX AI Insights</h1>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Load DAX data
    dax_df = load_dax_data_from_csv()
    
    if not dax_df.empty:
        dax_df = dax_df.dropna(subset=['Date']).sort_values("Date")
        
        # Create two columns: chart on left, metrics on right
        chart_col, metrics_col = st.columns([2, 1])
        
        with chart_col:
            st.markdown('<h3 class="section-title">📈 DAX Performance (Past Month)</h3>', unsafe_allow_html=True)
            
            # Create candlestick chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=dax_df['Date'], 
                open=dax_df['Open'],
                high=dax_df['High'], 
                low=dax_df['Low'],
                close=dax_df['Close'],
                increasing_line_color='#10B981',
                decreasing_line_color='#EF4444',
                name="DAX"
            ))
            
            fig.update_layout(
                xaxis_title='Date', 
                yaxis_title='Price (EUR)',
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                height=500,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E5E7EB'),
                xaxis=dict(gridcolor='rgba(59, 130, 246, 0.2)'),
                yaxis=dict(gridcolor='rgba(59, 130, 246, 0.2)')
            )
            fig.update_xaxes(type='date')
            st.plotly_chart(fig, use_container_width=True)
        
        with metrics_col:
            st.markdown('<h3 class="section-title">📊 Current DAX Values</h3>', unsafe_allow_html=True)
            
            # Get latest values
            latest_row = dax_df.iloc[-1]
            previous_row = dax_df.iloc[-2] if len(dax_df) > 1 else latest_row
            
            # Current price
            current_price = latest_row['Close']
            change, change_pct = calculate_change(current_price, previous_row['Close'])
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">{current_price:.2f} EUR</div>
                <div class="{'change-positive' if change >= 0 else 'change-negative'}">
                    {'+' if change >= 0 else ''}{change:.2f} ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Today's High
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Today's High</div>
                <div class="metric-value">{latest_row['High']:.2f} EUR</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Today's Low
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Today's Low</div>
                <div class="metric-value">{latest_row['Low']:.2f} EUR</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Opening Price
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Opening Price</div>
                <div class="metric-value">{latest_row['Open']:.2f} EUR</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Last Updated
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Updated</div>
                <div style="color: #9CA3AF; font-weight: 500;">{latest_row['Date'].strftime('%B %d, %Y')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Predict Tomorrow Button
            if st.button("🔮 Predict Tomorrow's DAX Value", key="predict_btn"):
                st.session_state.show_prediction = True
                st.session_state.page = "prediction"
                st.rerun()
    
    else:
        st.error("No DAX data available. Please check the data source.")

# =======================================================================================
# 🚀 PREDICTION PAGE
# =======================================================================================
elif st.session_state.page == "prediction":
    # Back button
    if st.button("← Back to Dashboard", key="back_btn"):
        st.session_state.page = "home"
        st.session_state.show_prediction = False
        st.session_state.show_news = False
        st.session_state.show_explanation = False
        st.rerun()
    
    st.markdown('<h1 class="main-title">🤖 DAX AI Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Show prediction with animation
    with st.spinner("🤖 Running LSTM model and analyzing sentiment..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Faster animation
            progress_bar.progress(i + 1)
        progress_bar.empty()
    
    # Main layout: prediction on left, buttons on right
    pred_col, button_col = st.columns([2, 1])
    
    with pred_col:
        # Main prediction display
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">Tomorrow's DAX Prediction</div>
            <div class="prediction-value">{next_price:.2f} EUR</div>
            <div class="prediction-date">Forecast for {pred_date.date()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with button_col:
        st.markdown('<h3 class="section-title">📊 Analysis Options</h3>', unsafe_allow_html=True)
        
        # Top 10 News Button
        if st.button("📰 Show Top 10 News", key="news_btn"):
            st.session_state.show_news = not st.session_state.show_news
            st.session_state.show_explanation = False
            st.rerun()
        
        # LLM Explanation Button
        if st.button("🧠 LLM Explanation", key="explanation_btn"):
            st.session_state.show_explanation = not st.session_state.show_explanation
            st.session_state.show_news = False
            st.rerun()
    
    # Content display area
    if st.session_state.show_news:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">📰 Top 10 Impactful News Headlines</h3>', unsafe_allow_html=True)
        
        # Get unique headlines
        unique_headlines = get_unique_headlines(top_headlines, 10)
        
        # Display unique news items
        for i, (_, row) in enumerate(unique_headlines.iterrows()):
            clean_headline_text = clean_headline(row['headline'])
            
            # Skip if headline is empty after cleaning
            if not clean_headline_text.strip():
                continue
            
            st.markdown(f"""
            <div class="news-item">
                <span class="news-number">{i+1}.</span>
                <a href='{row['url']}' target='_blank'>
                    {clean_headline_text}
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.show_explanation:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">🧠 LLM-Based Model Explanation</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="analysis-section">
            {explanation}
        </div>
        """, unsafe_allow_html=True)