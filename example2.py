import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title
st.title("Stock Prediction")

# Stock selection
stocks = ("QQQ","GLD", "AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# Prediction years
n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text("Loading data...")
try:
    data = load_data(selected_stock)
    data_load_state.text("Loading finished!")
    
    data.dropna(inplace=True) #Drop rows with missing values
    data.drop_duplicates(inplace=True)

    #Handle outliers. Remove where close price is outside 3 standard deviations from the mean
    data = data[np.abs(data["Close"] - data["Close"].mean()) <= (3 * data["Close"].std())]

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Display raw data
st.subheader("Raw data")
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Stock Open"))
    figure.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Stock Close"))
    figure.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_raw_data()

# Prepare data for regression
data['DateOrdinal'] = data['Date'].map(lambda x: x.toordinal())  # Convert dates to ordinal for regression
X = data[['DateOrdinal']].values
y = data['Close'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display performance metrics
st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Root Mean Squared Error: {rmse:.2f}")

# Plot test vs predicted
st.subheader("Test vs Predicted")
def plot_test_vs_predicted():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=X_test.flatten(), y=y_test, mode='markers', name='Test'))
    figure.add_trace(go.Scatter(x=X_test.flatten(), y=y_pred, mode='lines', name='Predicted'))
    figure.layout.update(title_text="Test vs Predicted", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(figure)

plot_test_vs_predicted()

# Forecast with Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model_prophet = Prophet()

# Fit the Prophet model
with st.spinner("Training the Prophet model..."):
    try:
        model_prophet.fit(df_train)
    except Exception as e:
        st.error(f"Error training Prophet model: {e}")
        st.stop()

# Make future predictions with Prophet
future = model_prophet.make_future_dataframe(periods=period)
forecast = model_prophet.predict(future)

# Display forecast data
st.subheader("Forecast Data")
st.write(forecast.tail())

# Plot forecast
st.subheader("Forecast Plot")
try:
    forecast_figure = plot_plotly(model_prophet, forecast)
    st.plotly_chart(forecast_figure)
except Exception as e:
    st.error(f"Error plotting forecast: {e}")

# Display forecast components
st.subheader("Forecast Components")
try:
    forecast_components = model_prophet.plot_components(forecast)
    st.write(forecast_components)
except Exception as e:
    st.error(f"Error plotting forecast components: {e}")
