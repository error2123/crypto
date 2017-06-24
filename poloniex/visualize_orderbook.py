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


while(True):
    dates = []
    for x in xrange(int(sys.argv[2])):
        dates.append("{}".format(currency_pair, (datetime.date.today() - datetime.timedelta(x)).strftime('%Y-%m-%d')))

    