# rsi vs msi comparison

from api import *
import sys
import config
import time
import logging
import datetime
import pandas
import time
from redis_client import batch_read_from_cache

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
from plotly import tools


# cmd line to run. argv[1] is the lookback
#python ema_bot_read_from_redis_v2.py 3

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)



# Window length for moving average
rsi_window_length = 14

# bin limits for the histogram @todo change it when playing with a bigger amount
negative = -500
positive = 500

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

def render_plotly_ema(df, rsi_list, rsi_orders, chart_name='candlestick_and_ema'):
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
    trace_rsi_ewma = go.Scatter(x=dates, y=rsi_list[0].values, name='rsi_ewma')
    trace_rsi_sma = go.Scatter(x=dates, y=rsi_list[1].values, name='rsi_sma')
    trace_30_line = go.Scatter(x=dates, y=[30 for x in dates], name='rsi=30')
    trace_70_line = go.Scatter(x=dates, y=[70 for x in dates], name='rsi_70')
    trace_buys = go.Scatter(x=[x[1] for x in rsi_orders[0]], y=[x[0] for x in rsi_orders[0]], mode='markers',
                            name='buys', marker=dict(color='00FF00', line=dict(width=2)))
    trace_sells = go.Scatter(x=[x[1] for x in rsi_orders[1]], y=[x[0] for x in rsi_orders[1]], mode='markers',
                            name='sells',  marker=dict(color='FF0000', line=dict(width=2)))

    trace_rsi_val = go.Scatter(x=[x[1] for x in rsi_orders[2]], y=[x[0] for x in rsi_orders[2]], mode='markers',
                             name='rsi_val', marker=dict(color='000000', line=dict(width=2)))

    trace_buys_mfi = go.Scatter(x=[x[1] for x in rsi_orders[4]], y=[x[0] for x in rsi_orders[0]], mode='markers',
                            name='buys', marker=dict(color='551a8b', line=dict(width=2)))
    trace_sells_mfi = go.Scatter(x=[x[1] for x in rsi_orders[5]], y=[x[0] for x in rsi_orders[1]], mode='markers',
                             name='sells', marker=dict(color='ffa500', line=dict(width=2)))

    trace_rsi_val_mfi = go.Scatter(x=[x[1] for x in rsi_orders[6]], y=[x[0] for x in rsi_orders[2]], mode='markers',
                               name='rsi_val', marker=dict(color='000000', line=dict(width=2)))

    data = [trace, trace_buys, trace_sells, trace_buys_mfi, trace_sells_mfi]
    #data = [trace, trace_3, trace_5, trace_7, trace_10, trace_12,
    #        trace_15, trace_30, trace_35, trace_40, trace_45, trace_50,
    #        trace_60, trace_buys, trace_sells]
    data_rsi = [trace_rsi_ewma, trace_rsi_sma, trace_30_line, trace_70_line, trace_rsi_val, trace_rsi_val_mfi]
    #offline.plot(data, filename='candlestick_and_ema', auto_open=False)

    fig = tools.make_subplots(rows=4, cols=1)
    [fig.append_trace(d, 1, 1) for d in data]
    [fig.append_trace(d, 2, 1) for d in data_rsi]
    fig.append_trace(go.Histogram(x=rsi_orders[3], name="rsi", autobinx=False, xbins=dict(start=negative, end=positive,size=5)), 3, 1)
    fig.append_trace(go.Histogram(x=rsi_orders[7], name="mfi",autobinx=False, xbins=dict(start=negative, end=positive,size=5)), 4, 1)
    offline.plot(fig, filename=chart_name, auto_open=False)


def find_equilibrium_points(df):
    df = df.sort_values(['date'])
    list_index = []
    list_var = []
    for x in df.index:
        list_index.append(x)
        list_var.append(abs(max(df.loc[x].loc["ema_3"], df.loc[x].loc["ema_5"],
                        df.loc[x].loc["ema_7"], df.loc[x].loc["ema_10"],
                        df.loc[x].loc["ema_12"], df.loc[x].loc["ema_15"]) -
                        min(df.loc[x].loc["ema_3"], df.loc[x].loc["ema_5"],
                        df.loc[x].loc["ema_7"], df.loc[x].loc["ema_10"],
                        df.loc[x].loc["ema_12"], df.loc[x].loc["ema_15"])))

    # @todo we always assume its expanding for the first element
    # is this correct ?
    list_sign_change = [1]
    for y in xrange(1, len(list_var)):
        if list_var[y-1] - list_var[y] > 0:
            list_sign_change.append(1)
        else:
            list_sign_change.append(-1)

    equilibrium_points = [list_index[l] for l in xrange(1, len(list_sign_change)) if list_sign_change[l-1] + list_sign_change[l] == 0]
    print equilibrium_points

    breakout_candidates = [idx for idx in xrange(len(list_var)) if 3 < list_var[idx] > 10]

    buy_or_sell_at = []

    for z in breakout_candidates:
        for b in xrange(z+1, len(list_var)):
            # if the variance is not increasing give up and move to the next point
            if list_var[b] - list_var[b-1] < 0:
                break
            else:
                if list_var[b] > 30:
                    buy_or_sell_at.append(list_index[b])
                    break
    print buy_or_sell_at



import matplotlib.pyplot as plt


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

def compute_mfi(df):
    # implemented from here
    #https://www.metatrader5.com/en/terminal/help/indicators/volume_indicators/mfi

    tp = ((df['close'] + df['low'] + df['high'])/3) * df['volume']

    # Get the difference in price from previous step
    delta = tp.diff()
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
        if entered_buy_zone:
            # if we are still in the zone and rsi starts increasing, BUY IT!
            # or if we stepping outside the zone buy it as well
            if (rsi_sma[indexes[x]] < min and rsi_sma[indexes[x]] > previous) or rsi_sma[indexes[x]] >= min:
                bought_at, bought_time = df.loc[indexes[x]].loc['close'], indexes[x]
                buys.append((bought_at, bought_time))
                bought = True
                rsi_l.append((rsi_sma[indexes[x]], bought_time))
                entered_buy_zone = False
        elif entered_sell_zone:
            if (rsi_sma[indexes[x]] > max and rsi_sma[indexes[x] < previous]) or rsi_sma[indexes[x]] <= max:
                sold_at, sold_time = df.loc[indexes[x]].loc['close'], indexes[x]
                sells.append((sold_at, sold_time))
                bought = False
                rsi_l.append((rsi_sma[indexes[x]], sold_time))
                total_profit += sold_at - bought_at
                pnl.append(sold_at - bought_at)
                print("Bought at {} on {}, Sold at {} on {} Profit/loss {}".format(bought_at, bought_time, sold_at, sold_time, sold_at-bought_at))
                entered_sell_zone = False
        elif rsi_sma[indexes[x]] <= min and previous > min and bought is False:
            entered_buy_zone = True
        elif rsi_sma[indexes[x]] >= max and previous < max and bought is True:
            entered_sell_zone = True
        previous = rsi_sma.loc[indexes[x]]

    print ("Total Profit: {}".format(total_profit))
    return buys, sells, rsi_l, pnl


while(True):
    dates = []
    for x in xrange(int(sys.argv[1])):
        dates.append((datetime.date.today() - datetime.timedelta(x)).strftime('%Y-%m-%d'))
    logger.info("Getting Datapoints for folllowing dates: {}".format(dates))
    full_chart = batch_read_from_cache(dates)
    df = pandas.DataFrame.from_dict(full_chart, orient="index")
    logger.info("DF: {}".format(df))
    df = df.sort_values(['date'])
    guppy_ema_calculation(df)
    render_plotly_chart(df)
    rsi_ewma, rsi_sma = compute_rsi(df)
    mfi_ewma, mfi_sma = compute_mfi(df)

    print "Profit based on MFI_SMA, 40, 70"
    mfi_buys, mfi_sells, mfi_val, mfi_pnl = compute_buy_sell_from_rsi(mfi_sma, 40, 70)

    print "No. of total MFI_SMA trades {}".format(len(mfi_val))
    ## print rsi_ewma, rsi_sma
    #render_plotly_ema(df, [mfi_ewma, mfi_sma], [mfi_buys, mfi_sells, mfi_val], "mfi_sma.html")

    print "Profit based on MFI_EWMA, 50, 60"
    emfi_buys, emfi_sells, emfi_val, emfi_pnl = compute_buy_sell_from_rsi(mfi_ewma, 50, 60)

    print "No. of total MFI_EWMA trades {}".format(len(emfi_val))
    # print rsi_ewma, rsi_sma
    #render_plotly_ema(df, [mfi_ewma, mfi_sma], [mfi_buys, mfi_sells, mfi_val], "mfi_ema.html")


    print "Profit based on RSI_SMA, 30, 70"
    rsi_buys, rsi_sells, rsi_val, rsi_pnl = compute_buy_sell_from_rsi(rsi_sma, 30, 70)

    print "No. of total RSI_SMA trades {}".format(len(rsi_val))

    render_plotly_ema(df, [mfi_sma, rsi_sma], [rsi_buys,
                                               rsi_sells,
                                               rsi_val,
                                               rsi_pnl,
                                               mfi_buys,
                                               mfi_sells,
                                               mfi_val,
                                               mfi_pnl], "rsi_vs_mfi_sma.html")


    #print "Profit based on RSI_SMA_WITH_STOP_LOSS"
    #rsi_buys, rsi_sells, rsi_val = compute_buy_sell_from_rsi_v2(rsi_sma, 30, 70)


    print "Profit based on RSI_EMA, 40, 70"
    rsi_buys, rsi_sells, rsi_val, rsi_pnl = compute_buy_sell_from_rsi(rsi_ewma, 40, 70)

    print "No. of total RSI_EMA trades {}".format(len(rsi_val))
    # print rsi_ewma, rsi_sma
    render_plotly_ema(df, [rsi_ewma, mfi_ewma], [rsi_buys, rsi_sells, rsi_val, rsi_pnl,
                                                 emfi_buys, emfi_sells, emfi_val, emfi_pnl], "rsi_vs_mfi_ema.html")

    find_equilibrium_points(df)
    time.sleep(3 * 60)


