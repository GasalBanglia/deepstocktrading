import gradio as gr
# from transformers import pipeline
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from sklearn.preprocessing import MinMaxScaler
from models import *
# from gradio_rangeslider import RangeSlider


def createScaler(ticker):
    stock_data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    features = stock_data[['Open', 'Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(features)
    return scaler

# Create a scaler based on the original training data.
scaler = createScaler("NVDA")


# Define sequence data creation function
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def download_stock_data(ticker, start=None, end=None, period="1y"):
    if start and end:
        stock_data = yf.download(ticker, start=start, end=end)
    elif period:
        stock_data = yf.download(ticker, period=period)
    else: 
        raise ValueError("Either start and end dates or period must be provided.")
    return stock_data

def get_transformed_features(df, ticker):
    features = df[['Open', 'Close']]
    scaler = createScaler(ticker)
    features_scaled = scaler.transform(features)
    return features_scaled, scaler

def get_predictions(model, input, seq_length):

    model.eval()
    
    X, actuals = create_sequences(input, seq_length)
    X = torch.Tensor(X)
    print("Getting Predictions...")
    with torch.no_grad():
        final_predictions, predicted_sequences = model(X)
    
    print(f"Type of final_predictions output from {type(model)}:", type(final_predictions))
    try:    
        final_predictions = final_predictions.detach().numpy()
        predicted_sequences = predicted_sequences.detach().numpy()
    except Exception as e:
        print(f"An error occurred while detaching tensors: {e}")

    return final_predictions, actuals


def predict(model_name="rnn2", ticker="NVDA", start="2023-01-01", end="2023-12-31"):

    stock_data = download_stock_data(ticker, start, end)
    if stock_data.empty:
        print("No data found for the given ticker or date range.")
        return None

    

    model = torch.load(f"models/{model_name}_model.pth", weights_only=False)

    seq_length = 30
    input, scaler = get_transformed_features(df=stock_data, ticker=ticker)
    predictions, actuals = get_predictions(model, input, seq_length)
    
    actuals = scaler.inverse_transform(actuals)
    actuals_aligned = np.empty((len(stock_data), 2))
    actuals_aligned[:] = np.nan
    actuals_aligned[seq_length:] = actuals

    # Ensure the predictions and actuals are aligned with the original data's dates
    predictions = scaler.inverse_transform(predictions)
    predictions_aligned = np.empty((len(stock_data), 2))
    predictions_aligned[:] = np.nan
    predictions_aligned[seq_length:] = predictions

    # Adjust the DataFrame to have a column indicating whether the value is actual or predicted
    df_actual = pd.DataFrame({
        "Date": stock_data.index,
        # "Closing Price": stock_data["Close"].values.squeeze(),
        # "Closing Price": actuals_aligned[:, 1],
        "Closing Price": actuals_aligned[:, 1],
        "Type": "Actual"
    })

    # predictions_aligned = scaler.inverse_transform(predictions_aligned)

    df_predicted = pd.DataFrame({
        "Date": stock_data.index,  # Align the dates with the predictions
        # "Closing Price": predictions_aligned[:, 1],
        "Closing Price": predictions_aligned[:, 1],
        "Type": "Predicted"
    })

    # Concatenate the actual and predicted DataFrames
    df_combined = pd.concat([df_actual, df_predicted])

    return df_combined



# Create a DataFrame with the dates, actual close prices, and predicted close prices
# df = pd.DataFrame({
#     "Date": stock_data.index,
#     "Actual Close": stock_data["Close"].values.squeeze(),
#     "Predicted Close": predictions_aligned[:, 1]
# })

# Extrapolate predictions n_days into the future, using 'leadup' number of days to train the hidden state.
# @model: trained model
# @ticker: stock ticker symbol
# @forecast_date: date to forecast, in YYYY-MM-DD format
# @leadup: number of days prior to forecast date to train the hidden state
# @n_days: number of days into the future to forecast
def forecastNDays(model_name, ticker, forecast_start, leadup=30, forecast_length=10):
    # Load the trained model
    # Recognize that it gets reloaded every time this function is run, which 
    # means the hidden state should be zeroed out
    model = torch.load(f"models/{model_name}_model.pth", weights_only=False)
    
    # adjust the forecast start date by "leadup" number of days.
    leadup_start = pd.to_datetime(forecast_start) - pd.Timedelta(days=leadup)
    forecast_end = pd.to_datetime(forecast_start) + pd.Timedelta(days=forecast_length)

    # Download stock data for the leadup period
    stock_data_raw = download_stock_data(ticker, start=leadup_start, end=forecast_end)
    if stock_data_raw.empty:
        print("No data found for the given ticker or date range.")
        return None

    # Extract the relevant columns and rows
    stock_data = stock_data_raw[["Open", "Close"]].iloc[0:leadup].copy()

    # Scale the data    
    features_scaled, scaler = get_transformed_features(df=stock_data, ticker=ticker)

    # Use the model to forecast the stock prices.
    model.eval()
    forecast = []
    for i in range(forecast_length):
        # Get the last 30 days of data
        input = features_scaled[i:i + leadup]
        input = torch.Tensor([input]) # Create the batch dimension.
        # print(f"input at i={i}: {input}")
        with torch.no_grad():
            prediction, _ = model(input)

        # print(f"prediction: {prediction}, input: {input}")
        prediction = prediction.detach().numpy().squeeze() # Get rid of the batch dimension
        forecast.append(prediction) # Get rid of the batch dimension
        features_scaled = np.vstack([features_scaled, prediction]) # Append the prediction to the input data.
    # forecast = np.array(forecast).squeeze() # Get rid of the batch dimension
    forecast = scaler.inverse_transform(forecast)
    

    # Create a DataFrame with the forecasted values
    forecast_dates = pd.date_range(start=forecast_start, periods=forecast_length)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Close": forecast[:, 1],
        "Type": "Forecasted"
    })
    actual_df = pd.DataFrame({
        "Date": stock_data_raw.index,
        "Close": stock_data_raw["Close"].values.squeeze(),
        "Type": "Actual"
    })
    df_forecasted = pd.concat([forecast_df, actual_df])
    return df_forecasted


df_combined = predict("rnn2", "NVDA", "2023-01-02", "2023-12-31")
df_forecasted = forecastNDays("rnn2", "NVDA", "2023-01-02", leadup=30, forecast_length=10)

text = "## The range is: {min} to {max}"

# Drop rows with NaN values
# df.dropna(inplace=True)

with gr.Blocks(theme="ocean") as demo:
    gr.Markdown("## Stock Price Prediction")
    gr.Markdown("### Predicting stock prices using a recurrent neural network, with 30-day leadup to single-day forecasting.")
    with gr.Row():
        with gr.Column():
            ticker = gr.Textbox(label="Ticker", value="NVDA")
            gr.Examples(label="Popular Stocks", examples=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"], inputs=ticker)
        with gr.Column():
            start_date = gr.Textbox(label="Start Date", value="2023-01-02")
            end_date = gr.Textbox(label="End Date", value="2023-12-31")
        with gr.Column():
            model = gr.Dropdown(label="Model", choices=["rnn2", "ltc2", "ltc3", "ltc4", "ltc5", "rnn_untrained"], value="rnn2")
            btn = gr.Button("Predict") 
    # with gr.Row():
        
    #     range_slider = RangeSlider(minimum=0, maximum=24*365, value=(23*365, 24*365))
    #     range_ = gr.Markdown(value=text.format(min=0, max=365*24), label="Days since 2000-01-01") # 365 days * 24 years
    #     range_slider.change(
    #         lambda s: text.format(min=(pd.to_datetime("2000-01-01") + pd.to_timedelta(s[0], unit='D')).strftime('%Y-%m-%d'), 
    #               max=(pd.to_datetime("2000-01-01") + pd.to_timedelta(s[1], unit='D')).strftime('%Y-%m-%d')), 
    #         range_slider, range_,
    #         show_progress="hide", trigger_mode="always_last"
    #     )
    plot = gr.LinePlot(df_combined, x="Date", y="Closing Price", color="Type", title="Predicted vs Actual (30-day Leadup)")
    
    gr.Markdown("## Forecasting")
    gr.Markdown("### Forecasting stock prices using a recurrent neural network, with variable-day leadup.")
    with gr.Row():
        # Create the lineplot to display the model's forecasting ability.
        forecast_start = gr.Textbox(label="Forecast Start Date", value="2023-01-02")
        leadup = gr.Number(label="Leadup", value=30)
        forecast_length = gr.Number(label="Forecast Length", value=10) 
        forecast_btn = gr.Button("Forecast")

    
    
    
    forecast_plot = gr.LinePlot(df_forecasted, x="Date", y="Close", color="Type", title=f"Forecasted vs Actual ({leadup.value}-day Leadup)")
    

    btn.click(predict, inputs=[model, ticker, start_date, end_date], outputs=plot)
    forecast_btn.click(forecastNDays, inputs=[model, ticker, forecast_start, leadup, forecast_length], outputs=forecast_plot)

demo.launch()