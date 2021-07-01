import pandas_datareader.data as web
from datetime import date,timedelta

import plotly.express as px
import plotly.graph_objects as go

import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import keras.backend as K

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

default_height = 600
default_width = 800
    
end = date.today()
start = end - timedelta(365)

model_folder = __file__
model_folder = os.path.dirname(model_folder)
model_path = os.path.join(model_folder,'models')

ticker_list = ['AAPL','MSFT','AMZN','GOOG','TSLA','FB','NVDA','PYPL','NFLX','CMCSA']
model_dict = {}

for i in ticker_list:
    model_dict.update({i:load_model(os.path.join(model_path,i+'.h5'))})

def close_price(stock):

    df = web.DataReader(stock, 'yahoo', start, end)
    df.reset_index(inplace = True)

    
    fig = px.line(df, x="Date", y="Close", title=f'Closing Price of {stock}')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[{'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 1, 'label': '1y', 'step': 'year', 'stepmode': 'backward'}]
        )
    )


    graph = fig.to_html(full_html=False, default_height=default_height, default_width=default_width)
    return graph

def candle_chart(stock):

    df = web.DataReader(stock, 'yahoo', start, end)
    df.reset_index(inplace = True)

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'])])

    fig.update_layout(title=f'Candle Chart for {stock}')

    fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=[{'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 1, 'label': '1y', 'step': 'year', 'stepmode': 'backward'}]
            )
        )

    graph = fig.to_html(full_html=False, default_height=default_height, default_width=default_width)
    return graph

def predicted_price(stock,n_days = 60):
    model_folder = __file__
    model_folder = os.path.dirname(model_folder)
    model_path = os.path.join(model_folder,'models',f'{stock}.h5')
    if os.path.exists(model_path):

        end = date.today()
        start = end - timedelta(n_days)
        df = web.DataReader(stock, 'yahoo', start, end)
        df.reset_index(inplace = True)
        df = df[['Close']]

        scaler = MinMaxScaler(feature_range=(0,1))
        df = scaler.fit_transform(df)

        df = df.reshape((1,*df.shape))
        model = model_dict[stock]
        prediction = model.predict(df)
        prediction = scaler.inverse_transform(prediction)[0][0]
        return round(prediction,2)
    else:
        return f'No model present for {stock}'