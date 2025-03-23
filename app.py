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

#visualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#split data
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
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

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original price')
plt.plot(y_predict,'r',label='Predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)