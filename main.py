import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('Stock Market Predictions')
start=st.text_input('Enter the Starting Date','2020-01-01')
end=st.text_input('Enter the Ending Date','2021-12-07')

user_input=st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)

st.subheader('Data from 2020-2021')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with moving avg 100 & 200')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig1=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200)
plt.plot(df.Close)
plt.legend()
st.pyplot(fig1)

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)

X_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    X_train.append(data_training_array[i-100:i, 0])
    y_train.append(data_training_array[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model=load_model('keras_model.h5')

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader('Prediction vs Orignal')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, color = 'red', label = 'Orignal Price')
plt.plot(y_predicted, color = 'green', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

shortema=df.Close.ewm(span=5,adjust=False).mean()
middleema=df.Close.ewm(span=20,adjust=False).mean()
longema=df.Close.ewm(span=50,adjust=False).mean()


st.subheader('Exponential Moving Average')
st.text("Calculating Exponential Moving Average after 100 Days ")
fig3=plt.figure(figsize=(14,6))
plt.title("3 EMA with closed price")
plt.plot(df['Close'],label='Orignal Price')
plt.plot(shortema,label='Short EMA')
plt.plot(middleema,label='Middle EMA')
plt.plot(longema,label='Long EMA')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
st.pyplot(fig3)

df['short']=shortema
df['middle']=middleema
df['long']=longema


def buy_sell_functon(data):
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False

    for i in range(0, len(data)):
        if data['middle'][i] < data['long'][i] and data['short'][i] < data['middle'][
            i] and flag_long == False and flag_short == False:
            buy_list.append(data['Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and data['short'][i] > data['middle'][i]:
            sell_list.append(data['Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['middle'][i] > data['long'][i] and data['short'][i] > data['middle'][
            i] and flag_long == False and flag_short == False:
            buy_list.append(data['Close'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_long == True and data['short'][i] < data['middle'][i]:
            sell_list.append(data['Close'][i])
            buy_list.append(np.nan)
            flag_short = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    return (buy_list, sell_list)

df['buy'] = buy_sell_functon(df)[0]
df['sell'] = buy_sell_functon(df)[1]

st.subheader('Buy and Sell Indicators')
st.text("### Using LSTM Model Prediction of these stocks is Done ")
fig4=plt.figure(figsize=(15,7))
plt.title('Buy and Sell price',fontsize=18)
plt.plot(df['Close'],label='Close Price',color='blue',alpha=0.30)
plt.plot(shortema,label='short EMA',color='red',alpha=0.30)
plt.plot(middleema,label='middle EMA',color='orange',alpha=0.30)
plt.plot(longema,label='long EMA',color='green',alpha=0.30)
plt.scatter(df.index, df['buy'],color='black', marker='*',alpha=1,label='Buy')
plt.scatter(df.index, df['sell'],color='green', marker='^',alpha=1,label='Sell')
plt.xlabel('Date')
plt.ylabel('close price')
plt.legend()
st.pyplot(fig4)
