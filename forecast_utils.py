### forecast_utils.py (rewritten with better modularization and clarity)

import os
import re
import time
import requests
import pandas as pd
import feedparser
import nltk
import spacy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from difflib import get_close_matches
from newspaper import Article
from dotenv import load_dotenv
from prophet import Prophet
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import root_mean_squared_error

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Setup and Initialization
load_dotenv()
nltk.download('vader_lexicon')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

marketstack_api_key = os.getenv('MARKETSTACK_API_KEY')
recommendation_log = []

# Utility Functions

def clean_company_name(name):
    name = name.upper()
    name = re.sub(r'\b(LISTED ON.*|THERE IS ONLY.*|THE ONLY PUBLIC.*)\b', '', name)
    name = re.sub(r'\b(LTD|LIMITED|CORPORATION|CORP|INC|PVT|PRIVATE|PLC)\b', '', name)
    name = re.sub(r'[^A-Z &]', '', name)
    name = name.replace('&', 'AND').strip()
    return name

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    return 'neutral'

def extract_companies_from_text(text):
    doc = nlp(text)
    return list(set(ent.text.strip() for ent in doc.ents if ent.label_ == 'ORG'))

def match_companies_to_symbols(companies, symbol_df):
    matched_symbols = []
    for name in companies:
        match = get_close_matches(name.upper(), symbol_df['Name'].str.upper().tolist(), n=1, cutoff=0.6)
        if match:
            row = symbol_df[symbol_df['Name'].str.upper() == match[0]]
            if not row.empty:
                matched_symbols.append(row.iloc[0]['Symbol'])
    return matched_symbols


def add_technical_indicators(df):
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

    df['Volatility'] = df['Close'].rolling(window=10).std()
    return df


def calculate_strategy_metrics(df):
    df['Cumulative Returns'] = (1 + df['Return']).cumprod()
    sharpe_ratio = df['Return'].mean() / df['Return'].std() * np.sqrt(252)
    cumulative_max = df['Cumulative Returns'].cummax()
    drawdown = df['Cumulative Returns'] / cumulative_max - 1
    max_drawdown = drawdown.min()
    return sharpe_ratio, max_drawdown

# Article Analysis
def analyze_article(url, symbols_df):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text[:1000]
    except Exception as e:
        print(f"❌ Failed to load article: {e}")
        return "Failed to load article", "neutral", [], []

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, max_tokens=512)
    summary = llm.invoke(f"Summarize this article:\n{text}").content
    sentiment = analyze_sentiment(summary)

    company_prompt = f"List publicly traded companies mentioned in this article:\n{text}"
    companies_text = llm.invoke(company_prompt).content
    raw_names = [clean_company_name(line.strip()) for line in companies_text.split('\n') if len(line.strip()) > 3]
    stock_names = list(set(raw_names))

    # Clean symbols_df
    symbols_df['CleanName'] = symbols_df['Name'].str.upper()
    symbols_df['CleanName'] = symbols_df['CleanName'].str.replace(r'\b(LTD|LIMITED|CORPORATION|PVT|INC|PLC)\b', '', regex=True)
    symbols_df['CleanName'] = symbols_df['CleanName'].str.replace('&', 'AND').str.strip()

    matched = []
    for name in stock_names:
        exact = symbols_df[symbols_df['CleanName'] == name]['Symbol'].tolist()
        if exact:
            matched.extend(exact)
        else:
            close = get_close_matches(name, symbols_df['CleanName'], n=1, cutoff=0.85)
            if close:
                symbol = symbols_df[symbols_df['CleanName'] == close[0]]['Symbol'].values[0]
                matched.append(symbol)

    return summary, sentiment, stock_names, list(set(matched))

# News Sources

def get_articles():
    base_url = "https://www.cnbctv18.com/market/stocks/"
    headers = {"User-Agent": "Mozilla/5.0"}
    titles, urls = [], []
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('a', class_='jsx-95506e352219bddb story-media')
        for a in articles:
            try:
                titles.append(a['title'].strip())
                urls.append(urljoin(base_url, a['href']))
            except:
                continue
    except Exception as e:
        print(f"Error fetching CNBC articles: {e}")
    return titles, urls

def get_articles_moneycontrol():
    feed_url = "https://www.moneycontrol.com/rss/MCtopnews.xml"
    titles, urls = [], []
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:
            titles.append(entry.title)
            urls.append(entry.link)
    except Exception as e:
        print(f"Error fetching Moneycontrol feed: {e}")
    return titles, urls

# Forecasting

def generate_recommendation(sentiment, forecast_change):
    if forecast_change > 0 and sentiment == 'positive':
        return "Buy"
    elif forecast_change < 0 and sentiment == 'negative':
        return "Sell"
    return "Hold"

def log_recommendation(symbol, sentiment, direction, recommendation, date):
    recommendation_log.append({
        "Date": date,
        "Stock": symbol,
        "Sentiment": sentiment,
        "Forecast Direction": direction,
        "Recommendation": recommendation,
        
    })

def get_marketstack_data(symbol, api_key, start_date, end_date):
    url = "http://api.marketstack.com/v1/eod"
    params = {
        'access_key': api_key,
        'symbols': symbol,
        'date_from': start_date,
        'date_to': end_date,
        'limit': 1000
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.rename(columns={
            'date': 'Date', 'open': 'Open', 'close': 'Close',
            'high': 'High', 'low': 'Low', 'volume': 'Volume'
        }, inplace=True)
        return df[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def forecast_stock(symbol, n_future_days=30, export, sentiment):
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = get_marketstack_data(symbol, marketstack_api_key, start_date, end_date)

    if df.empty:
        df = get_marketstack_data(symbol + ".XNSE", marketstack_api_key, start_date, end_date)

    if df.empty or len(df) < 100:
        print(f"Not enough data for {symbol}.")
        return [], None

    try:
        df.sort_values('Date', inplace=True)
        df = add_technical_indicators(df)
        df.dropna(inplace=True)
        df['ds'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['y'] = df['Close']

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        for col in ['Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal', 'Volatility']:
            model.add_regressor(col)

        model.fit(df[['ds', 'y', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal', 'Volatility']])

        future = model.make_future_dataframe(periods=n_future_days)
        extra = df[['ds', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal', 'Volatility']]
        future = pd.merge(future, extra, on='ds', how='left').ffill()

        forecast = model.predict(future)

        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        actuals = df[['ds', 'y']].rename(columns={'ds': 'Date', 'y': 'Actual'})
        merged_df = pd.merge(forecast_df, actuals, on='Date', how='left')
        merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')

        forecast_change = merged_df['Forecast'].iloc[-1] - merged_df['Forecast'].iloc[-n_future_days]
        direction = "up" if forecast_change > 0 else "down"
        recommendation = generate_recommendation(sentiment, forecast_change)
        log_recommendation(symbol, sentiment, direction, recommendation, merged_df['Date'].iloc[-1])

        if export:
            merged_df.to_csv(f"forecast_{symbol}_{n_future_days}_days.csv", index=False)

        return merged_df['Forecast'].tolist(), merged_df

    except Exception as e:
        print(f"Forecasting failed for {symbol}: {e}")
        return [], None


def plot_portfolio(df_bt):
    df_bt['Date'] = pd.to_datetime(df_bt['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(df_bt['Date'], df_bt['Portfolio Value'], label='Portfolio Value')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def backtest_stock(symbol, sentiment='neutral', window_size=60, backtest_days=90, hold_days=3, stop_loss=0.05, take_profit=0.10):
    df = get_marketstack_data(symbol, marketstack_api_key, "2015-01-01", datetime.now().strftime("%Y-%m-%d"))

    if df.empty:
        df = get_marketstack_data(symbol + '.XNSE', marketstack_api_key, "2015-01-01", datetime.now().strftime("%Y-%m-%d"))

    if df.empty or len(df) < window_size + backtest_days:
        print(f"❌ Not enough data for backtesting {symbol}")
        return pd.DataFrame()

    df.sort_values('Date', inplace=True)
    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df['ds'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['y'] = df['Close']

    results = []
    capital = 100000
    position = 0
    entry_day = -1
    entry_price = 0
    returns = []

    for i in range(len(df) - window_size - backtest_days, len(df) - window_size):
        train = df.iloc[i:i+window_size].copy()
        test = df.iloc[i+window_size:i+window_size+1].copy()
        test_date = test.iloc[0]['ds']
        actual = test.iloc[0]['y']

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        for col in ['Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal', 'Volatility']:
            model.add_regressor(col)

        try:
            model.fit(train[['ds', 'y', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal', 'Volatility']])
            future = test[['ds', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'Signal', 'Volatility']]
            forecast = model.predict(future)

            forecasted = forecast.iloc[0]['yhat']
            change = forecasted - train.iloc[-1]['y']
            forecast_dir = "up" if change > 0 else "down"
            actual_dir = "up" if actual - train.iloc[-1]['y'] > 0 else "down"
            correct = forecast_dir == actual_dir

            rec = generate_recommendation(sentiment, change)
            rec_correct = (
                (rec == "Buy" and actual_dir == "up") or
                (rec == "Sell" and actual_dir == "down") or
                (rec == "Hold")
            )

            # Apply stop-loss/take-profit
            if position > 0:
                change_pct = (actual - entry_price) / entry_price
                if change_pct <= -stop_loss or change_pct >= take_profit or i - entry_day >= hold_days:
                    capital = position * actual
                    position = 0
                    entry_day = -1
                    entry_price = 0

            # Execute trade
            buy_price = train.iloc[-1]['y']
            if rec == "Buy" and capital > 0:
                position = capital / buy_price
                capital = 0
                entry_day = i
                entry_price = buy_price

            portfolio_value = capital + (position * actual)
            pct_return = (portfolio_value - 100000) / 100000
            returns.append(pct_return)

            results.append({
                "Date": test_date.strftime('%Y-%m-%d'),
                "Forecast": forecasted,
                "Actual": actual,
                "Correct Direction": correct,
                "Recommendation": rec,
                "Correct Recommendation": rec_correct,
                "Portfolio Value": portfolio_value,
                "Return": pct_return
            })

        except Exception as e:
            print(f"⚠️ Backtest error at {test_date}: {e}")

    df_bt = pd.DataFrame(results)
    if not df_bt.empty:
        mae = mean_absolute_error(df_bt['Actual'], df_bt['Forecast'])
        rmse = sqrt(mean_squared_error(df_bt['Actual'], df_bt['Forecast']))
        df_bt['MAE'] = mae
        df_bt['RMSE'] = rmse *len(df_bt)
        df_bt['Return'] = df_bt['Return'].fillna(0)
        sharpe, max_dd = calculate_strategy_metrics(df_bt)
        df_bt['Sharpe Ratio'] = sharpe
        df_bt['Max Drawdown'] = max_dd
        # Plot in Streamlit
        import streamlit as st
        st.line_chart(df_bt.set_index('Date')['Portfolio Value'], use_container_width=True)

        # Export to CSV
        csv = df_bt.to_csv(index=False).encode('utf-8')
        st.download_button("📤 Download Backtest CSV", csv, "backtest_results.csv", "text/csv")

        # Show stats summary
        st.metric("Final Portfolio Value", f"₹{df_bt['Portfolio Value'].iloc[-1]:,.2f}")
        st.metric("Total Return", f"{df_bt['Return'].iloc[-1]*100:.2f}%")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Max Drawdown", f"{max_dd:.2%}")

        # Stop-loss / take-profit flag visualization
        stop_loss_days = df_bt[df_bt['Recommendation'] == 'Buy'][df_bt['Return'].diff() < -0.05]
        take_profit_days = df_bt[df_bt['Recommendation'] == 'Buy'][df_bt['Return'].diff() > 0.1]

        if not stop_loss_days.empty:
            st.write("🔻 **Stop-loss triggered on:**")
            st.dataframe(stop_loss_days[['Date', 'Portfolio Value', 'Return']])

        if not take_profit_days.empty:
            st.write("🟢 **Take-profit triggered on:**")
            st.dataframe(take_profit_days[['Date', 'Portfolio Value', 'Return']])

    return df_bt

   
