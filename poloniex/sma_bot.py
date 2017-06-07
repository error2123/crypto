from api import *
import sys
import config
import time
import logging
import datetime
import pandas

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

LOOKBACK_SEC = 120 * 60
INTERVAL = 60 * 5


def guppy_ema_calculation(df):
    df["ema_3"] = pandas.ewma(df["close"], span=3)
    df["ema_5"] = pandas.ewma(df["close"], span=5)
    df["ema_7"] = pandas.ewma(df["close"], span=7)
    df["ema_10"] = pandas.ewma(df["close"], span=10)
    df["ema_12"] = pandas.ewma(df["close"], span=12)
    df["ema_15"] = pandas.ewma(df["close"], span=15)
    df["ema_30"] = pandas.ewma(df["close"], span=30)
    df["ema_35"] = pandas.ewma(df["close"], span=35)
    df["ema_40"] = pandas.ewma(df["close"], span=40)
    df["ema_45"] = pandas.ewma(df["close"], span=45)
    df["ema_50"] = pandas.ewma(df["close"], span=50)
    df["ema_60"] = pandas.ewma(df["close"], span=60)


def render_chart(df):
    # Making plot
    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=6, colspan=1)

    # Converts raw mdate numbers to dates
    ax1.xaxis_date()
    plt.xlabel("Date")
    print(df)
    df = df.reset_index()
    df = df.sort_values(['date'])
    data_4plt = [[mdates.date2num(datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) for x in df['index'].values], df['open'].values, df['high'].values, df['low'].values, df['close'].values]


    #import ipdb;
    #ipdb.set_trace()
    # Making candlestick plot
    candlestick_ohlc(ax1, data_4plt, width=1, colorup='g', colordown='r', alpha=0.75)
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def render_plotly_chart(df):
    #import ipdb;
    #ipdb.set_trace()
    trace = go.Candlestick(x=[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.index],
                           open=df['open'].values,
                           high=df['high'].values,
                           low=df['low'].values,
                           close=df['close'].values)
    data = [trace]
    offline.plot(data, filename='styled_candlestick', auto_open=False)

def render_plotly_ema(df):
    dates = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.index]
    trace_3 = go.Scatter(
        x=dates,
        y=df['ema_3'],
        name='ema_3',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2)
    )
    trace_5 = go.Scatter(
        x=dates,
        y=df['ema_5'],
        name='ema_5',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2)
    )
    trace_7 = go.Scatter(
        x=dates,
        y=df['ema_7'],
        name='ema_7',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2)
    )
    trace_10 = go.Scatter(
        x=dates,
        y=df['ema_10'],
        name='ema_10',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2)
    )
    trace_12 = go.Scatter(
        x=dates,
        y=df['ema_12'],
        name='ema_12',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2)
    )
    trace_15 = go.Scatter(
        x=dates,
        y=df['ema_15'],
        name='ema_15',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2)
    )
    trace_30 = go.Scatter(
        x=dates,
        y=df['ema_30'],
        name='ema_30',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )
    trace_35 = go.Scatter(
        x=dates,
        y=df['ema_35'],
        name='ema_35',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )
    trace_40 = go.Scatter(
        x=dates,
        y=df['ema_40'],
        name='ema_40',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )
    trace_45 = go.Scatter(
        x=dates,
        y=df['ema_45'],
        name='ema_45',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )
    trace_50 = go.Scatter(
        x=dates,
        y=df['ema_50'],
        name='ema_50',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )
    trace_60 = go.Scatter(
        x=dates,
        y=df['ema_60'],
        name='ema_60',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=2)
    )

    trace = go.Candlestick(x=[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.index],
                           open=df['open'].values,
                           high=df['high'].values,
                           low=df['low'].values,
                           close=df['close'].values)
    data = [trace, trace_3, trace_5, trace_7, trace_10, trace_12, trace_15, trace_30, trace_35, trace_40, trace_45, trace_50, trace_60]
    offline.plot(data, filename='candlestick_and_ema', auto_open=False)


t_end = time.time()
t_start = t_end - LOOKBACK_SEC
P = poloniex(APIKey=config.api_key, Secret=config.secret_key)
resp = P.returnChartData(currencyPair="USDT_BTC", start=t_start, end=t_end, period=INTERVAL)
full_chart = {'{}'.format(datetime.datetime.fromtimestamp(int(r["date"])).strftime('%Y-%m-%d %H:%M:%S')): r for r in resp}
df = pandas.DataFrame.from_dict(full_chart, orient="index")
logger.info(df)
#import ipdb; ipdb.set_trace()
#df.index = pandas.to_datetime(df.pop('timestamp'))
guppy_ema_calculation(df)
render_plotly_chart(df)
render_plotly_ema(df)
logger.info(df)

while(True):
    logger.info("Start time: {} End time: {}".format(t_start, t_end))
    resp = P.returnChartData(currencyPair="USDT_BTC", start=t_start, end=t_end, period=INTERVAL)
    resp = {'{}'.format(datetime.datetime.fromtimestamp(int(r["date"])).strftime('%Y-%m-%d %H:%M:%S')): r for r in resp}
    full_chart = dict(full_chart.items() + resp.items())
    df = pandas.DataFrame.from_dict(full_chart, orient="index")
    guppy_ema_calculation(df)
    render_plotly_chart(df)
    render_plotly_ema(df)
    logger.info("RESP: {}, DF: {}".format(resp, df))
    t_start = t_end + 1
    time.sleep(INTERVAL)
    # commenting this because going off of the latest time is more robust as it will
    # pick more data points if it slept for sometime
    #t_end = t_start + INTERVAL
    t_end = time.time()

