from api import *
import sys
import config
import time
import logging
import datetime
import pandas
import time
from redis_client import batch_read_from_cache
import numpy as np
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools
import plotly.figure_factory as ff



# cmd line to run. argv[1] is the lookback
#python rsi_ema.py 3

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Window length for moving average
rsi_window_length = 14
currency_pair = "USDT_BTC"

#current candlestick period
candle_period = 5


# bin limits for the histogram @todo change it when playing with a bigger amount
negative = -500
positive = 500

# ewma upper and lower bound
lb = 20
ub = 70


def render_plotly(df, rsi, rsi_orders, lower_bound, upper_bound, chart_name='candlestick'):
    dates = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.index]
    trace = go.Candlestick(x=[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.index],
                           open=df['open'].values,
                           high=df['high'].values,
                           low=df['low'].values,
                           close=df['close'].values)
    trace_rsi_ewma = go.Scatter(x=dates, y=rsi.values, name='rsi_ewma')
    trace_lb = go.Scatter(x=dates, y=[lower_bound for x in dates], name='rsi={}'.format(lower_bound))
    trace_ub = go.Scatter(x=dates, y=[upper_bound for x in dates], name='rsi={}'.format(upper_bound))
    trace_buys = go.Scatter(x=[x[1] for x in rsi_orders[0]], y=[x[0] for x in rsi_orders[0]], mode='markers',
                            name='buys', marker=dict(color='00FF00', line=dict(width=2)))
    trace_sells = go.Scatter(x=[x[1] for x in rsi_orders[1]], y=[x[0] for x in rsi_orders[1]], mode='markers',
                            name='sells',  marker=dict(color='FF0000', line=dict(width=2)))
    trace_rsi_val = go.Scatter(x=[x[1] for x in rsi_orders[2]], y=[x[0] for x in rsi_orders[2]], mode='markers',
                             name='rsi_val', marker=dict(color='000000', line=dict(width=2)))
    trace_volume = go.Scatter(x=dates, y=[float(df.loc[x].loc['volume']) for x in df.index], name='volume')
    avg_vol = np.mean(df["volume"])
    trace_avg_volume = go.Scatter(x=dates, y=[avg_vol for x in dates], name='avg_volume')
    trace_mavg_volume = go.Scatter(x=dates, y=[float(df.loc[x].loc['moving_avg_vol']) for x in df.index], name='avg_volume')
    
    pnl = rsi_orders[3]
    stats_dict = {}
    stats_dict["Mean"] =  np.mean(pnl)
    stats_dict["Median"] =  np.median(pnl)
    stats_dict["Highest_gain"] =  max(pnl)
    stats_dict["Highest_loss"] =  min(pnl)
    stats_dict["Total_trades"] =  len(pnl)
    stats_dict["Negative_trades"] =  len([x for x in pnl if x <= 0])
    stats_dict["Negative_trades_percentage"] =  len([x for x in pnl if x <= 0])/len(pnl)
    print stats_dict
    #stats_dict["Bought_sold"] = "{}".format([
    perf = pandas.DataFrame.from_dict({"performance": stats_dict}, orient="columns").to_html()
    with open("charts/rsi_ema_vanilla/rsi_ema_vanilla_volume_performance.html", "w") as fw:
        fw.write(perf)
    
    data = [trace, trace_buys, trace_sells]
    data_rsi = [trace_rsi_ewma, trace_lb, trace_ub, trace_rsi_val]
    data_volume = [trace_volume, trace_avg_volume, trace_mavg_volume]
    fig = tools.make_subplots(rows=4, cols=1)
    [fig.append_trace(d, 1, 1) for d in data]
    [fig.append_trace(d, 3, 1) for d in data_rsi]
    [fig.append_trace(d, 4, 1) for d in data_volume]
    
    fig.append_trace(go.Histogram(x=rsi_orders[3], name="rsi", autobinx=False, xbins=dict(start=negative, end=positive,size=5)), 2, 1)
    offline.plot(fig, filename=chart_name, auto_open=False)



def add_avg_volume_24hr(df):
    volume_tracker = deque(maxlen=24 * 60 /5)
    volume_init = 150000
    [volume_tracker.append(volume_init) for x in xrange(0, 24 * 60 /5)]
    avg_volume = []
    for x in df.index:
        volume_tracker.append(df.loc[x].loc['volume'])
        avg_volume.append(sum(volume_tracker)/len(volume_tracker))
    avg_v_s = pandas.Series(avg_volume, df.index)
    df['moving_avg_vol'] = avg_v_s

def compute_rsi(df):

    # Get the difference in price from previous step
    delta = df['close'].diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = pandas.stats.moments.ewma(up, rsi_window_length)
    roll_down1 = pandas.stats.moments.ewma(down.abs(), rsi_window_length)

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    roll_up2 = pandas.rolling_mean(up, rsi_window_length)
    roll_down2 = pandas.rolling_mean(down.abs(), rsi_window_length)

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    return RSI1, RSI2


def compute_buy_sell_from_rsi(rsi_sma, min, max):
    indexes = [x for x in rsi_sma.index][rsi_window_length-1:]
    previous = rsi_sma[indexes[0]]
    bought = False
    buys = []
    sells = []
    rsi_l = []
    pnl =[]
    total_profit = 0
    entered_buy_zone = False
    entered_sell_zone = False
    for x in xrange(1, len(indexes)):
            # if we are still in the zone and rsi starts increasing, BUY IT!
            # or if we stepping outside the zone buy it as well
            #if ((rsi_sma[indexes[x]] < min and rsi_sma[indexes[x]] > previous) or rsi_sma[indexes[x]] >= min) and df.loc[indexes[x]].loc['volume'] > 8 * df.loc[indexes[x]].loc['moving_avg_vol']:
        if rsi_sma[indexes[x]] < min and df.loc[indexes[x]].loc['volume'] > 2 * df.loc[indexes[x]].loc['moving_avg_vol'] and bought is False:
                bought_at, bought_time = df.loc[indexes[x]].loc['close'], indexes[x]
                buys.append((bought_at, bought_time))
                bought = True
                rsi_l.append((rsi_sma[indexes[x]], bought_time))
                entered_buy_zone = False
        
            #if ((rsi_sma[indexes[x]] > max and rsi_sma[indexes[x] < previous]) or rsi_sma[indexes[x]] <= max) and df.loc[indexes[x]].loc['volume'] < df.loc[indexes[x]].loc['moving_avg_vol']/2:
        elif rsi_sma[indexes[x]] > max and df.loc[indexes[x]].loc['volume'] < df.loc[indexes[x]].loc['moving_avg_vol'] and bought is True:
                sold_at, sold_time = df.loc[indexes[x]].loc['close'], indexes[x]
                sells.append((sold_at, sold_time))
                bought = False
                rsi_l.append((rsi_sma[indexes[x]], sold_time))
                total_profit += sold_at - bought_at
                pnl.append(sold_at - bought_at)
                print("Bought at {} on {}, Sold at {} on {} Profit/loss {}".format(bought_at, bought_time, sold_at, sold_time, sold_at-bought_at))
                entered_sell_zone = False
        
        previous = rsi_sma.loc[indexes[x]]

    print ("Total Profit: {}".format(total_profit))
    return buys, sells, rsi_l, pnl


while(True):
    dates = []
    for x in xrange(int(sys.argv[1])):
        dates.append("{}###{}".format(currency_pair, (datetime.date.today() - datetime.timedelta(x)).strftime('%Y-%m-%d')))
    logger.info("Getting Datapoints for folllowing dates: {}".format(dates))
    full_chart = batch_read_from_cache(dates)
    full_chart = {key.split("###")[1]:full_chart[key]  for key in full_chart}
    df = pandas.DataFrame.from_dict(full_chart, orient="index")
    logger.info("DF: {}".format(df))
    df = df.sort_values(['date'])
    add_avg_volume_24hr(df)
    rsi_ewma, rsi_sma = compute_rsi(df)
    rsi_buys, rsi_sells, rsi_val, rsi_pnl = compute_buy_sell_from_rsi(rsi_ewma, lb, ub)
    
    render_plotly(df, rsi_ewma, [rsi_buys, rsi_sells, rsi_val, rsi_pnl], lb, ub, "charts/rsi_ema_vanilla/rsi_ema_vanilla_volume.html")
    
    time.sleep(3 * 60)


