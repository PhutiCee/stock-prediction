import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

# Input field for the ticker symbol
st.title("Financial Asset Data Viewer")
ticker = st.text_input("Enter the ticker symbol of the asset:", "GLD")

# Fetch the data for the specified ticker
asset = yf.Ticker(ticker)
asset_history = asset.history(period="max")

# Set up the Streamlit app
st.subheader(f"{ticker} Data")
st.write(asset_history.tail())

# Plot the closing prices
fig, ax = plt.subplots()
asset_history.plot.line(y="Close", use_index=True, ax=ax)
ax.set_title(f'{ticker} Closing Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Closing Price')
st.pyplot(fig)

# Clean up the data
del asset_history["Dividends"]
del asset_history["Stock Splits"]

# Add target columns for modeling
asset_history["Tomorrow"] = asset_history["Close"].shift(-1)
asset_history["Target"] = (asset_history["Tomorrow"] > asset_history["Close"]).astype(int)

# Display the cleaned data
st.write("Data after removing Dividends and Stock Splits columns:")
st.write(asset_history.tail())

# Filter the data from a specific date
asset_filtered = asset_history.loc["1990-01-01":].copy()

# Split the data into train and test sets
train = asset_filtered.iloc[:-100]
test = asset_filtered.iloc[-100:]

# Specify the predictors
predictors = ["Close", "Volume", "High", "Low"]

# Correlation Matrix
st.subheader("Correlation Matrix")
correlation_matrix = asset_filtered[predictors + ["Target"]].corr()
st.write(correlation_matrix)

# Scatter Plots
st.subheader("Scatter Plots")
for predictor in predictors:
    fig, ax = plt.subplots()
    ax.scatter(asset_filtered[predictor], asset_filtered["Close"], alpha=0.5)
    ax.set_title(f'Scatter Plot: {predictor} vs Close')
    ax.set_xlabel(predictor)
    ax.set_ylabel('Close')
    st.pyplot(fig)

# Train the model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(train[predictors], train["Target"])

# Make predictions
prediction = model.predict(test[predictors])
prediction = pd.Series(prediction, index=test.index)

# Display the precision score
st.subheader("Precision Score")
precision_score_value = precision_score(test["Target"], prediction)
st.write(precision_score_value)

# Feature Importances
feature_importances = pd.Series(model.feature_importances_, index=predictors)
st.subheader("Feature Importances")
st.bar_chart(feature_importances)

# Display feature importances in detail
st.write("Feature Importances Details:")
st.write(feature_importances)

# Optionally refine the model using the most significant predictors
mean_importance = feature_importances.mean()
significant_predictors = feature_importances[feature_importances > mean_importance].index.tolist()
st.write("Significant Predictors:", significant_predictors)

if len(significant_predictors) > 0:
    model.fit(train[significant_predictors], train["Target"])
    refined_prediction = model.predict(test[significant_predictors])
    refined_precision_score_value = precision_score(test["Target"], refined_prediction)
    st.subheader("Refined Precision Score with Significant Predictors")
    st.write(refined_precision_score_value)
else:
    st.write("No significant predictors identified based on the mean feature importance.")
