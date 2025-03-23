import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2014-01-01'
end = '2024-12-31'
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker','META')
df = yf.download(user_input,start=start,end=end)
st.subheader('Data from 2014-2024')
st.write(df.describe())

#dropdown
column_options = ['Open', 'High', 'Low', 'Close', 'Volume']  # Add more if needed
selected_column = st.selectbox("Select the column to visualize:", column_options)

#visualizations
#Closing price
st.subheader(f'{selected_column} Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df[selected_column], 'b', label=selected_column)
plt.legend()
st.pyplot(fig)

st.subheader(f'{selected_column} Price vs Time Chart with 100MA')
ma100=df[selected_column].rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='100 Moving Average')
plt.plot(df[selected_column], 'b',label=selected_column)
plt.legend()
st.pyplot(fig)

st.subheader(f'{selected_column} Price vs Time Chart with 200MA')
ma100=df[selected_column].rolling(100).mean()
ma200=df[selected_column].rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='100 Moving Average')
plt.plot(ma200,'g',label='200 Moving Average')
plt.plot(df[selected_column], 'b',label=selected_column)
plt.legend()
st.pyplot(fig)

#split data
data_training=pd.DataFrame(df[selected_column][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df[selected_column][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_arr=scaler.fit_transform(data_training)


#load my model
model=load_model('keras_model.h5')
#testing
past_data=data_training.tail(100)
final_df = pd.concat([past_data, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

#predictions 
y_predict=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predict=y_predict*scale_factor
y_test=y_test*scale_factor

#final graph

st.subheader(f'Predictions vs Original for {selected_column}')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original price')
plt.plot(y_predict,'r',label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#explain

st.subheader("Need Help Understanding the Graphs?")

if st.button("Explain Trends"):
    if selected_column == "Close":
        st.write("""
        - The **Closing Price** represents the last traded price of the stock each day.
        - The **Moving Averages (100MA & 200MA)** smooth out short-term price fluctuations to highlight longer-term trends.
        - If the **100MA crosses above the 200MA**, it suggests a **bullish trend (uptrend)**. If it falls below, it's a **bearish trend (downtrend)**.
        - Predictions help analyze if the stock follows a stable pattern or fluctuates unexpectedly.
        """)
    elif selected_column == "Open":
        st.write("""
        - The **Opening Price** shows the first price traded each day.
        - Comparing this with the closing price helps understand **daily volatility**.
        - Large differences between open and close can indicate **high market activity**.
        """)
    elif selected_column == "Volume":
        st.write("""
        - **Volume** represents the number of shares traded.
        - High volume with price increase suggests **strong buying interest**.
        - Low volume with price increase may indicate **weak momentum**.
        """)
    else:
        st.write("""
        - The **High & Low Prices** show the highest and lowest points the stock reached during a period.
        - If the high keeps increasing, the stock may be gaining strength.
        - If the low is continuously dropping, it might be in a **downtrend**.
        """)
