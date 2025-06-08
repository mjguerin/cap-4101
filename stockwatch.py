"""
StockWatch - A Comprehensive Stock Portfolio Management Application
CAP4104 - Human-Computer Interaction
Author: Marc Guerin

This application provides a complete stock trading and portfolio management interface
with real-time data integration, user location mapping, and interactive visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import csv
import os
import requests

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CAP4104 - [ StockWatch ]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize persistent session variables for navigation and data storage
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = []

# ============================================================================
# API CONFIGURATION AND CONSTANTS
# ============================================================================

# Finnhub API configuration for real-time stock data
FINNHUB_API_KEY = "d12asl1r01qmhi3hh660d12asl1r01qmhi3hh66g"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Popular stock symbols with company names for dropdown selections
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corporation',
    'TSLA': 'Tesla Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix Inc.',
    'AMD': 'Advanced Micro Devices',
    'INTC': 'Intel Corporation',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corporation',
    'IBM': 'IBM Corporation',
    'PYPL': 'PayPal Holdings',
    'ADBE': 'Adobe Inc.',
    'UBER': 'Uber Technologies',
    'SPOT': 'Spotify Technology',
    'ZOOM': 'Zoom Video Communications',
    'SQ': 'Block Inc.',
    'TWTR': 'Twitter Inc.'
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_page(page_name):
    """
    Navigation utility function to switch between application pages
    """
    st.session_state.current_page = page_name
    st.rerun()


def make_api_request(url):
    """
    Generic HTTP request handler using urllib for API calls
    Returns parsed JSON data or None on failure
    """
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        return None
    except json.JSONDecodeError:
        return None


# ============================================================================
# GEOLOCATION SERVICES
# ============================================================================

def get_user_location():
    """
    Retrieve user's geographic location using IP-based geolocation
    Falls back to Miami, FL coordinates if service unavailable
    """
    try:
        # Use ipinfo.io service for reliable IP geolocation
        response = requests.get('https://ipinfo.io/json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Parse latitude and longitude from "lat,lon" format
            lat, lon = data['loc'].split(',')
            return {
                'latitude': float(lat),
                'longitude': float(lon),
                'city': data.get('city', 'Miami'),
                'region': data.get('region', 'Florida'),
                'country': data.get('country', 'United States')
            }
    except Exception as e:
        st.warning(f"Could not determine location: {str(e)}")

    # Default fallback location (Miami, Florida)
    return {
        'latitude': 25.7617,
        'longitude': -80.1918,
        'city': 'Miami',
        'region': 'Florida',
        'country': 'United States'
    }


# ============================================================================
# STOCK DATA API FUNCTIONS
# ============================================================================

def get_stock_quote(symbol, api_key=FINNHUB_API_KEY):
    """
    Fetch real-time stock quote data from Finnhub API
    Returns formatted stock data dictionary or None on failure
    """
    try:
        url = f"{FINNHUB_BASE_URL}/quote?symbol={symbol}&token={api_key}"
        data = make_api_request(url)

        # Validate response and check for valid price data
        if data and 'c' in data and data['c'] != 0:
            return {
                'symbol': symbol,
                'current_price': round(data['c'], 2),
                'change': round(data['d'], 2),
                'percent_change': round(data['dp'], 2),
                'high': round(data['h'], 2),
                'low': round(data['l'], 2),
                'open': round(data['o'], 2),
                'previous_close': round(data['pc'], 2)
            }
        return None
    except Exception as e:
        return None


def get_company_profile(symbol, api_key=FINNHUB_API_KEY):
    """
    Retrieve company profile information from Finnhub API
    Falls back to predefined company data if API unavailable
    """
    try:
        url = f"{FINNHUB_BASE_URL}/stock/profile2?symbol={symbol}&token={api_key}"
        data = make_api_request(url)

        if data and data.get('name'):
            return data
        # Fallback to predefined company information
        return {
            'name': POPULAR_STOCKS.get(symbol, f"{symbol} Corp"),
            'ticker': symbol,
            'industry': 'Technology',
            'country': 'US'
        }
    except Exception:
        # Default company profile for unknown symbols
        return {
            'name': POPULAR_STOCKS.get(symbol, f"{symbol} Corp"),
            'ticker': symbol,
            'industry': 'Technology',
            'country': 'US'
        }


def get_market_news(api_key=FINNHUB_API_KEY):
    """
    Fetch latest market news from Finnhub API
    Returns list of news articles or empty list on failure
    """
    try:
        url = f"{FINNHUB_BASE_URL}/news?category=general&token={api_key}"
        data = make_api_request(url)

        if data and isinstance(data, list):
            return data[:10]  # Limit to 10 most recent articles
        return None
    except Exception:
        return []


# ============================================================================
# DATA MANAGEMENT FUNCTIONS
# ============================================================================

def load_transactions_from_csv():
    """
    Load transaction history from CSV file
    Converts CSV data to application-compatible format
    """
    try:
        df = pd.read_csv('transactions.csv')

        transactions = []
        for _, row in df.iterrows():
            transactions.append({
                'date': row['date'],
                'symbol': row['symbol'],
                'side': row['side'],
                'shares': int(row['shares']),
                'cost_price': float(row['cost_price']),
                'total': float(row['cost_price']) * int(row['shares'])
            })
        return transactions

    except Exception as e:
        st.warning(f"Could not load portfolio from CSV: {str(e)}. Using sample data.")
        return []


def load_portfolio_data():
    """
    Generate current portfolio positions from transaction history
    Calculates net shares and average cost basis for each symbol
    """
    if not st.session_state.transactions:
        return []

    # Aggregate transactions by symbol
    portfolio = {}
    for transaction in st.session_state.transactions:
        symbol = transaction['symbol']
        if symbol not in portfolio:
            portfolio[symbol] = {
                'symbol': symbol,
                'shares': 0,
                'total_cost': 0,
                'date_added': transaction['date'],
            }

        # Calculate net position based on buy/sell transactions
        if transaction['side'] == 'BUY':
            portfolio[symbol]['shares'] += transaction['shares']
            portfolio[symbol]['total_cost'] += transaction['total']
        elif transaction['side'] == 'SELL':
            portfolio[symbol]['shares'] -= transaction['shares']
            portfolio[symbol]['total_cost'] -= transaction['total']

    # Filter active positions and calculate average price
    portfolio_data = []
    for symbol, position in portfolio.items():
        if position['shares'] > 0:
            position['avg_price'] = position['total_cost'] / position['shares']
            portfolio_data.append(position)

    return portfolio_data


def get_portfolio_metrics():
    """
    Calculate comprehensive portfolio performance metrics
    Returns dictionary with current value, P&L, and other key metrics
    """
    if not st.session_state.portfolio_data:
        return {
            'total_value': 0, 'total_cost': 0, 'total_gain_loss': 0,
            'total_gain_loss_pct': 0, 'daily_change': 0, 'positions_count': 0
        }

    total_value = 0
    total_cost = 0
    daily_change = 0

    # Calculate metrics for each position
    for position in st.session_state.portfolio_data:
        current_data = get_stock_quote(position['symbol'])
        if current_data:
            current_price = current_data['current_price']
            shares = position['shares']
            avg_price = position['avg_price']

            # Calculate position values
            position_value = shares * current_price
            position_cost = shares * avg_price
            position_daily_change = shares * current_data['change']

            total_value += position_value
            total_cost += position_cost
            daily_change += position_daily_change

    # Calculate overall portfolio performance
    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0

    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_pct': total_gain_loss_pct,
        'daily_change': daily_change,
        'positions_count': len(st.session_state.portfolio_data)
    }


def add_transaction(symbol, action, shares, price):
    """
    Record new transaction to CSV file
    Validates input parameters and appends to transaction log
    """
    if shares <= 0 or price <= 0:
        raise ValueError("Invalid shares or price")

    filename = "transactions.csv"
    file_exists = os.path.isfile(filename)

    # Append transaction to CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Create header row if file doesn't exist
            writer.writerow(['Date', 'Symbol', 'Action', 'Shares', 'Price'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d'), symbol, action.upper(), shares, price])


# ============================================================================
# USER INTERFACE COMPONENTS
# ============================================================================

def render_sidebar():
    """
    Render navigation sidebar with page selection buttons
    Maintains active page state and provides intuitive navigation
    """
    st.sidebar.title("[ StockWatch ]")

    # Define application pages with navigation metadata
    pages = [
        {'name': 'Dashboard', 'icon_class': 'dashboard', 'key': 'Dashboard'},
        {'name': 'Market', 'icon_class': 'market', 'key': 'Market'},
        {'name': 'Trade', 'icon_class': 'trade', 'key': 'Trade'},
        {'name': 'Transactions', 'icon_class': 'transactions', 'key': 'Transactions'},
        {'name': 'News', 'icon_class': 'news', 'key': 'News'}
    ]

    # Create interactive navigation buttons
    for page in pages:
        is_active = st.session_state.current_page == page['name']
        col1, col2 = st.sidebar.columns([1, 4])

        with col1:
            # Placeholder for custom icons
            st.markdown(f'<div class="custom-icon {page["icon_class"]}"></div>', unsafe_allow_html=True)

        with col2:
            display_name = page['key'] if page['key'] == 'Overview' else page['name']
            if st.button(display_name, key=f"nav_{page['name']}", use_container_width=True):
                st.session_state.current_page = page['name']
                st.rerun()


# ============================================================================
# PAGE RENDERING FUNCTIONS
# ============================================================================

def render_dashboard():
    """
    Render main dashboard with portfolio overview, performance chart, and location map
    Provides comprehensive portfolio summary and key metrics
    """
    st.markdown('<h2 class="main-header">Dashboard</h2>', unsafe_allow_html=True)

    # Load data if not already cached in session
    if not st.session_state.transactions:
        st.session_state.transactions = load_transactions_from_csv()

    if not st.session_state.portfolio_data:
        st.session_state.portfolio_data = load_portfolio_data()

    # Portfolio metrics display section
    col1, col2, col3 = st.columns(3)
    metrics = get_portfolio_metrics()

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${metrics['total_value']:,.2f}",
            delta=f"{metrics['total_gain_loss_pct']:+.2f}%",
            border=True
        )

    with col2:
        st.metric(
            label="Daily P&L",
            value=f"${metrics['total_gain_loss']:+,.2f}",
            delta="1.8%",
            border=True
        )

    with col3:
        st.metric(
            label="Active Positions",
            value=str(len(st.session_state.portfolio_data)),
            delta="+2",
            border=True
        )

    st.markdown("---")

    # Portfolio performance visualization
    st.subheader("Portfolio Performance")
    if st.session_state.portfolio_data:
        # Generate simulated historical portfolio data
        dates = []
        values = []

        for i in range(30):
            date = datetime.now() - timedelta(days=30 - i - 1)
            daily_value = 0

            # Calculate portfolio value for each historical day
            for position in st.session_state.portfolio_data:
                base_price = position['avg_price']
                # Simulate price variation with normal distribution
                random_variation = np.random.normal(0, 0.02)
                historical_price = base_price * (1 + random_variation)
                daily_value += position['shares'] * historical_price

            dates.append(date)
            values.append(daily_value)

        # Display interactive line chart
        chart_data = pd.DataFrame({'Portfolio Value': values}, index=dates)
        st.line_chart(chart_data)

    st.markdown("---")

    # Geographic location visualization
    st.subheader("User Geographic Location")
    user_location = get_user_location()

    # Create map data for Streamlit map component
    map_data = pd.DataFrame({
        "lat": [user_location['latitude']],
        "lon": [user_location['longitude']]
    })

    # Display interactive map with user location
    st.map(map_data, zoom=10)


def render_market():
    """
    Render market overview with indices, stock performance grid, and search functionality
    Provides comprehensive market data visualization
    """
    st.markdown('<h2 class="main-header">Market Overview</h2>', unsafe_allow_html=True)

    # Market indices overview (simulated data for demonstration)
    col1, col2, col3, col4 = st.columns(4)
    indices = [
        ('S&P 500', 4500.25, 45.30, 1.02),
        ('NASDAQ', 14250.80, -85.20, -0.59),
        ('DOW', 35200.15, 125.75, 0.36),
        ('VIX', 18.45, -1.25, -6.35)
    ]

    # Display major market indices
    for i, (name, value, change, change_pct) in enumerate(indices):
        with [col1, col2, col3, col4][i]:
            st.metric(name, f"{value:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")

    st.markdown("---")

    # Stock search and filtering interface
    col1, col2 = st.columns([3, 1])
    with col2:
        search_symbol = st.text_input("Search Stock", placeholder="Enter symbol...")
    with col1:
        st.subheader("Stock Performance")

    # Fetch real-time market data with progress indicator
    market_data = []
    progress_placeholder = st.empty()

    stocks_to_show = list(POPULAR_STOCKS.keys())
    if search_symbol:
        stocks_to_show = [s for s in stocks_to_show if search_symbol.upper() in s]

    # Display loading progress for API calls
    with progress_placeholder:
        progress_bar = st.progress(0)
        for i, symbol in enumerate(stocks_to_show):
            stock_data = get_stock_quote(symbol)
            if stock_data:
                market_data.append({
                    'symbol': symbol,
                    'name': POPULAR_STOCKS[symbol],
                    'price': stock_data['current_price'],
                    'change': stock_data['change'],
                    'change_pct': stock_data['percent_change'],
                    'high': stock_data['high'],
                    'low': stock_data['low']
                })
            progress_bar.progress((i + 1) / len(stocks_to_show))

    progress_placeholder.empty()

    # Market statistics calculation and display
    if market_data:
        changes = [item['change_pct'] for item in market_data]
        avg_change = np.mean(changes)
        positive_stocks = sum(1 for change in changes if change > 0)
        negative_stocks = sum(1 for change in changes if change < 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market Average", f"{avg_change:+.2f}%")
        with col2:
            st.metric("Advancing", str(positive_stocks), delta=f"{positive_stocks - negative_stocks:+d}")
        with col3:
            st.metric("Declining", str(negative_stocks))
        with col4:
            volatility = np.std(changes)
            st.metric("Volatility", f"{volatility:.2f}%")

        # Stock performance grid with custom styling
        st.markdown("---")
        cols_per_row = 3
        rows = [market_data[i:i + cols_per_row] for i in range(0, len(market_data), cols_per_row)]

        for row in rows:
            cols = st.columns(cols_per_row)
            for i, stock in enumerate(row):
                with cols[i]:
                    # Color-coded performance indicators
                    change_color = "#10b981" if stock['change_pct'] >= 0 else "#ef4444"
                    st.markdown(f"""
                    <div class="stock-card">
                        <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">{stock['symbol']}</h4>
                        <p style="margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #6b7280;">{stock['name'][:30]}...</p>
                        <p style="margin: 0 0 0.5rem 0; font-size: 1.4rem; font-weight: bold; color: #1f2937;">${stock['price']:.2f}</p>
                        <p style="margin: 0; color: {change_color}; font-weight: 600;">{stock['change_pct']:+.2f}% (${stock['change']:+.2f})</p>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #6b7280;">
                            H: ${stock['high']:.2f} | L: ${stock['low']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


def render_trade():
    """
    Render trading interface with stock analysis and order placement
    Provides comprehensive trading functionality with validation
    """
    st.markdown('<h2 class="main-header">Trading Center</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    # Stock analysis section
    with col1:
        st.subheader("Stock Analysis")
        selected_symbol = st.selectbox(
            "Select Stock",
            options=list(POPULAR_STOCKS.keys()),
            format_func=lambda x: f"{x} - {POPULAR_STOCKS[x]}",
            key="trade_stock_select"
        )

        # Display real-time stock information
        stock_data = get_stock_quote(selected_symbol)
        company_profile = get_company_profile(selected_symbol)

        if stock_data:
            # Key metrics display
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Current Price", f"${stock_data['current_price']:.2f}")
            with col_info2:
                st.metric("Daily Change", f"${stock_data['change']:+.2f}", f"{stock_data['percent_change']:+.2f}%")
            with col_info3:
                st.metric("Day High", f"${stock_data['high']:.2f}")
            with col_info4:
                st.metric("Day Low", f"${stock_data['low']:.2f}")

            # Company information display
            if company_profile:
                st.markdown(f"""
                **Company:** {company_profile['name']}  
                **Industry:** {company_profile.get('industry', 'N/A')}  
                **Country:** {company_profile.get('country', 'N/A')}
                """)

    # Order placement section
    with col2:
        st.subheader("Place Order")

        with st.form("trading_form"):
            action = st.radio("Order Type", ["BUY", "SELL"], horizontal=True)

            col_shares, col_price = st.columns(2)

            with col_shares:
                shares = st.number_input("Quantity", min_value=1, value=1, step=1)

            with col_price:
                if stock_data:
                    current_price = stock_data['current_price']
                    order_type = st.selectbox("Price Type", ["Market", "Limit", "Stop"])

                    # Price input based on order type
                    if order_type == "Market":
                        price = current_price
                        st.write(f"Market Price: ${current_price:.2f}")
                    elif order_type == "Limit":
                        price = st.number_input("Limit Price", value=current_price, step=0.01, format="%.2f")
                    else:  # Stop order
                        price = st.number_input("Stop Price", value=current_price * 0.95, step=0.01, format="%.2f")

            # Order summary and validation
            if stock_data:
                total_value = shares * price

                st.markdown("---")
                st.markdown("**Order Summary**")
                st.write(f"**Symbol:** {selected_symbol}")
                st.write(f"**Action:** {action}")
                st.write(f"**Shares:** {shares:,}")
                st.write(f"**Price:** ${price:.2f}")
                st.write(f"**Total:** ${total_value:,.2f}")

                agree = st.checkbox("I agree to the above order")

                # Portfolio impact analysis
                if action == "BUY":
                    st.info(f"💰 This will cost ${total_value:,.2f}")
                else:
                    # Validate sell orders against current holdings
                    current_position = None
                    for pos in st.session_state.portfolio_data:
                        if pos['symbol'] == selected_symbol:
                            current_position = pos
                            break

                    if current_position and current_position['shares'] >= shares:
                        proceeds = total_value
                        cost_basis = current_position['avg_price'] * shares
                        profit_loss = proceeds - cost_basis
                        st.info(f"This will generate ${proceeds:,.2f} (P&L: ${profit_loss:+,.2f})")
                    elif current_position:
                        st.error(f"Insufficient shares (you own {current_position['shares']})")
                    else:
                        st.error("No position found to sell")

                # Order execution
                submitted = st.form_submit_button("Submit Order", type="primary", use_container_width=True)

                if agree and submitted:
                    try:
                        # Additional validation for sell orders
                        if action == "SELL":
                            current_position = None
                            for pos in st.session_state.portfolio_data:
                                if pos['symbol'] == selected_symbol:
                                    current_position = pos
                                    break

                            if not current_position:
                                st.error("Cannot sell - no position found")
                            elif current_position['shares'] < shares:
                                st.error(f"Cannot sell {shares} shares - you only own {current_position['shares']}")
                            else:
                                add_transaction(selected_symbol, action, shares, price)
                                st.success(
                                    f"Order executed: {action} {shares} shares of {selected_symbol} at ${price:.2f}")
                                time.sleep(2)
                                st.rerun()
                        else:
                            add_transaction(selected_symbol, action, shares, price)
                            st.success(f"Order executed: {action} {shares} shares of {selected_symbol} at ${price:.2f}")
                            time.sleep(2)
                            st.rerun()

                    except Exception as e:
                        st.error(f"Order failed: {str(e)}")
                elif submitted and not agree:
                    st.error("You must agree to placing this order")


def render_transactions():
    """
    Render transaction history with detailed view and pagination
    Displays comprehensive transaction log with current market values
    """
    st.markdown('<h2 class="main-header">Transactions</h2>', unsafe_allow_html=True)

    # Load latest transaction data
    st.session_state.transactions = load_transactions_from_csv()
    loaded_transactions = st.session_state.transactions

    st.subheader(f"Transaction Details ({len(loaded_transactions)} transactions)")

    if not loaded_transactions:
        st.info("No transactions found.")
        return

    # Prepare display data with current market prices
    display_data = []
    for transaction in loaded_transactions:
        current_data = get_stock_quote(transaction['symbol'])
        current_price = current_data['current_price'] if current_data else transaction['cost_price']
        company_profile = get_company_profile(transaction['symbol'])

        display_data.append({
            'Date': transaction['date'],
            'Stock': transaction['symbol'],
            'Company': company_profile['name'],
            'Action': transaction['side'],
            'Shares': f"{transaction['shares']:,}",
            'Price': f"${transaction['cost_price']:.2f}",
            'Current': f"${current_price:.2f}",
            'Total': f"${transaction['total']:,.2f}"
        })

    # Implement pagination for large transaction lists
    page_size = 20
    total_pages = (len(display_data) + page_size - 1) // page_size

    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1)) - 1
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(display_data))
        display_data = display_data[start_idx:end_idx]

    # Display transaction table
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_news():
    """
    Render financial news section with filtering and source attribution
    Provides curated market news with time-based filtering
    """
    st.markdown('<h2 class="main-header">News</h2>', unsafe_allow_html=True)

    # News filtering interface
    col1, col2 = st.columns(2)
    with col1:
        news_source = st.selectbox("Source", ["All Sources", "Reuters", "Bloomberg", "CNBC", "MarketWatch"])
    with col2:
        time_filter = st.selectbox("Time", ["All Time", "Last 24 Hours", "Last Week"])

    # Fetch and filter news data
    news_data = get_market_news()

    if news_data:
        # Apply time-based filtering
        if time_filter == "Last 24 Hours":
            cutoff = datetime.now() - timedelta(hours=24)
            news_data = [n for n in news_data if datetime.fromtimestamp(n['datetime']) > cutoff]
        elif time_filter == "Last Week":
            cutoff = datetime.now() - timedelta(days=7)
            news_data = [n for n in news_data if datetime.fromtimestamp(n['datetime']) > cutoff]

        # Apply source filtering
        if news_source != "All Sources":
            news_data = [n for n in news_data if news_source.lower() in n.get('source', '').lower()]

        st.subheader(f"Top Headlines ({len(news_data)} articles)")
        st.markdown("---")

        # Display news articles with formatted layout
        for i, news_item in enumerate(news_data):
            st.markdown(f"""
                <div class="news-card">
                    <p style="margin: 0 0 0.5rem 0; color: #4b5563; line-height: 1.5;"><a href="{news_item['url']}" target="_blank" style="text-decoration: none; color: #3b82f6;">{news_item['headline']}</a></p>
                    <p style="margin: 0 0 0.5rem 0; color: #4b5563; line-height: 1.5;">{news_item.get('summary', news_item['headline'])[:200]}...</p>
                    <p style="color: #666; font-size: 0.9rem; line-height: 1.5;">{datetime.fromtimestamp(news_item['datetime']).strftime('%Y-%m-%d %H:%M')} - {news_item.get('source', 'Unknown Source')}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")

    else:
        st.error("Unable to load market news at this time.")


# ============================================================================
# MAIN APPLICATION CONTROLLER
# ============================================================================

def main():
    """
    Main application controller that handles navigation and page rendering
    Coordinates between sidebar navigation and content display
    """
    # Render navigation sidebar
    render_sidebar()

    # Route to appropriate page based on session state
    if st.session_state.current_page == 'Dashboard':
        render_dashboard()
    elif st.session_state.current_page == 'Market':
        render_market()
    elif st.session_state.current_page == 'Trade':
        render_trade()
    elif st.session_state.current_page == 'Transactions':
        render_transactions()
    elif st.session_state.current_page == 'News':
        render_news()

    # Application footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        CAP4104 - StockWatch - Marc Guerin
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()