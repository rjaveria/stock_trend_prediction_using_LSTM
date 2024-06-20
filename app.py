
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model



start = '2014-01-01'
end = '2024-05-15'
stock = 'AMZN'

st.title("Stock Trend Prediction App")
user_input= st.text_input("Enter Company Name:","AMZN")
df= yf.download(user_input, start, end)
df.head()

# Describing data
st.subheader("10-Year Stock Data from 2014 to 2024")
st.write(df.describe())

#Graph Visualization
st.subheader("Closing Price vs Time Chart")
fig=plt.figure(figsize=(10,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')

# Calculating Moving Averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Plotting
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100-Day MA')
plt.plot(ma200, label='200-Day MA')
plt.plot(df.Close, label='Closing Price')
plt.legend()  # Add a legend to the plot

# Displaying the plot in Streamlit
st.pyplot(fig)

#Split the data into training and testing 
data_train = pd.DataFrame(df.Close[0: int(len(df)*0.70)])
data_test = pd.DataFrame(df.Close[int(len(df)*0.70): len(df)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)
x_train = []
y_train = []

for i in range(100, data_train_array.shape[0]):
    x_train.append(data_train_array[i-100:i])
    y_train.append(data_train_array[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Load the model
model= load_model("stock_price_prediction.keras")