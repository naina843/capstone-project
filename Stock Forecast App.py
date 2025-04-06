import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

from forecast_utils import (
    get_articles,
    get_articles_moneycontrol,
    forecast_stock,
    recommendation_log,
    analyze_article , backtest_stock
)
from langchain_groq import ChatGroq

# Page setup
st.set_page_config(page_title="📈 Stock Insights Dashboard", layout="wide")

# Session state initialization
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# Forecast View
if st.session_state['page'] == 'forecast':
    symbol = st.session_state.get('forecast_symbol', '')
    sentiment = st.session_state.get('forecast_sentiment', 'neutral')
    days = 30
    export = False

    st.title(f"🔮 Forecast for {symbol}")

    with st.spinner("Generating forecast..."):
        yhat, forecast_df = forecast_stock(symbol, n_future_days=days, export=export, sentiment=sentiment)

    if forecast_df is not None and not forecast_df.empty:
        st.success("Forecast complete ✅")
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_df['Date'], forecast_df['Actual'], label='Actual', color='blue')
        ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='green')
        ax.fill_between(forecast_df['Date'], forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='gray', alpha=0.3)
        ax.set_title(f"Forecast vs Actual for {symbol.upper()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.dataframe(forecast_df)
    else:
        st.warning("No forecast data available.")

    if st.button("⬅️ Back to News"):
        st.session_state['page'] = 'main'
        st.rerun()

# Main Dashboard
elif st.session_state['page'] == 'main':
    st.title("📊 Stock News & Forecasting Dashboard")
    menu = st.sidebar.radio("Navigate", ["📰 Market News", "📈 Forecast Stock", "🗂️ Recommendations Log"])#, "🔁 Backtest Forecasts"])

    # Market News Tab
    if menu == "📰 Market News":
        st.subheader("🗞️ LLM-Powered Article Summaries & Symbol Detection")
        source = st.selectbox("Select Source", ["CNBC TV18", "Moneycontrol"])

        if 'article_cache' not in st.session_state:
            st.session_state['article_cache'] = {}

        if 'last_source' not in st.session_state or st.session_state['last_source'] != source:
            st.session_state['fetched_articles'] = get_articles() if source == "CNBC TV18" else get_articles_moneycontrol()
            st.session_state['last_source'] = source

        titles, links = st.session_state['fetched_articles']
        st.info(f"✅ Fetched {len(titles)} articles from {source}.")

        try:
            symbols_df = pd.read_csv("merged_cleaned stock&symbols.csv")[['Name', 'Symbol']].dropna(subset=['Symbol'])
            symbols_df = symbols_df[symbols_df['Symbol'] != "#REF!"]
        except Exception as e:
            st.error("❌ Could not load symbol mapping file.")
            st.stop()

        # Sentiment filter dropdown
        st.markdown("### 🔍 Filter Articles")
        sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "neutral", "negative"])

        for title, url in zip(titles, links):
            try:
                if url in st.session_state['article_cache']:
                    result = st.session_state['article_cache'][url]
                else:
                    with st.spinner(f"Analyzing: {title[:60]}..."):
                        summary, sentiment, companies, symbols = analyze_article(url, symbols_df)
                    result = {
                        "summary": summary,
                        "sentiment": sentiment,
                        "companies": companies,
                        "symbols": symbols
                    }
                    st.session_state['article_cache'][url] = result

                if sentiment_filter != "All" and result['sentiment'] != sentiment_filter:
                    continue

                sentiment_color = {
                    "positive": "green",
                    "neutral": "orange",
                    "negative": "red"
                }.get(result['sentiment'], "black")
                sentiment_emoji = {"positive": "✅", "neutral": "⚠️", "negative": "❌"}.get(result['sentiment'], "❔")

                st.markdown(f"### 📰 [{title}]({url})")
                st.markdown(f"**Summary**: {result['summary']}")
                st.markdown(f"**Sentiment**: <span style='color:{sentiment_color}'>{sentiment_emoji} `{result['sentiment']}`</span>", unsafe_allow_html=True)
                st.markdown(f"**Companies**: {', '.join(result['companies']) or 'N/A'}")
                st.markdown(f"**Symbols**: {', '.join(result['symbols']) or 'N/A'}")

                if result['symbols']:
                    symbol_key = f"symbol_{hash(url)}"
                    button_key = f"forecast_{hash(url)}"
                    selected = st.selectbox("🔍 Select symbol", result['symbols'], key=symbol_key)

                    if st.button("📈 Forecast", key=button_key):
                        st.session_state['forecast_symbol'] = selected
                        st.session_state['forecast_sentiment'] = result['sentiment']
                        st.session_state['page'] = 'forecast'
                        st.rerun()
                else:
                    st.markdown("*No forecastable symbols found.*")

                with st.expander("💬 Ask a question about this article"):
                    if 'full_texts' not in st.session_state:
                        st.session_state['full_texts'] = {}
                    if url not in st.session_state['full_texts']:
                        try:
                            import newspaper
                            art = newspaper.Article(url)
                            art.download()
                            art.parse()
                            st.session_state['full_texts'][url] = art.text
                        except:
                            st.session_state['full_texts'][url] = result['summary']

                    user_question = st.text_input("Ask your question here:", key=f"q_{hash(url)}")
                    if st.button("Get Answer", key=f"a_{hash(url)}") and user_question:
                        with st.spinner("Thinking..."):
                            llm = ChatGroq(model="llama3-8b-8192", temperature=0.5, max_tokens=300)
                            article_text = st.session_state['full_texts'][url]
                            prompt = f"Answer this question based on the following article:\n\n{article_text}\n\nQ: {user_question}"
                            try:
                                response = llm.invoke(prompt).content
                                st.success("Answer:")
                                st.write(response)
                            except Exception as e:
                                st.error(f"❌ Failed to generate answer: {e}")

                st.markdown("---")

            except Exception as e:
                st.warning(f"❌ Could not analyze article: {e}")

    # Manual Forecast Tab
    elif menu == "📈 Forecast Stock":
        st.subheader("🔮 Forecast Individual Stocks")
        symbol = st.text_input("Enter Symbol (e.g., AAPL, YESBANK)")
        sentiment = st.selectbox("Select Sentiment", ["positive", "neutral", "negative"])
        days = st.slider("Days to Forecast", 7, 90, 30)
        export = st.checkbox("Export forecast as CSV")

        if st.button("Run Forecast"):
            if not symbol:
                st.error("Please enter a symbol.")
            else:
                with st.spinner("Forecasting..."):
                    yhat, forecast_df = forecast_stock(symbol.upper(), n_future_days=days, export=export, sentiment=sentiment)

                if forecast_df is not None and not forecast_df.empty:
                    st.success("Forecast complete ✅")
                    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(forecast_df['Date'], forecast_df['Actual'], label='Actual', color='blue')
                    ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='green')
                    ax.fill_between(forecast_df['Date'], forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='gray', alpha=0.3)
                    ax.set_title(f"Forecast for {symbol.upper()}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    ax.grid(True)

                    st.pyplot(fig)
                    st.dataframe(forecast_df)
                else:
                    st.warning("No data found for this symbol.")

    # Recommendations Log
    elif menu == "🗂️ Recommendations Log":
        st.subheader("📌 Recommendation Log")
        if recommendation_log:
            st.dataframe(pd.DataFrame(recommendation_log))
        else:
            st.info("No recommendations generated yet.")

          

#    elif menu == "🔁 Backtest Forecasts":
#       st.subheader("📊 Backtesting Engine")
#        symbol = st.text_input("Enter symbol to backtest", "YESBANK")
#        sentiment = st.selectbox("Sentiment Assumption", ["positive", "neutral", "negative"])
 #       if st.button("Run Backtest"):
  #          with st.spinner("Running backtest..."):
   #             df_bt = backtest_stock(symbol, sentiment=sentiment)
    #        if not df_bt.empty:
     #           st.success("Backtest Complete ✅")
      #          st.dataframe(df_bt)
       #         correct_pct = df_bt['Correct Direction'].mean() * 100
        #        st.metric("Direction Accuracy", f"{correct_pct:.2f}%")
         #       st.metric("MAE", f"{df_bt['MAE'].iloc[0]:.2f}")
          #      st.metric("RMSE", f"{df_bt['RMSE'].iloc[0]:.2f}")
           # else:
            #    st.warning("Backtest failed or not enough data.")
