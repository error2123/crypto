from api import *
import json
import logging
import sys
import config
import datetime
from redis_client import batch_write_to_cache

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

#LOOKBACK_SEC = 60 * 60 * 8
LOOKBACK_SEC = 60 * 60 * 48
INTERVAL = 60 * 5

# ok this will look convoluted, but what I am trying to do below is
# to expand my start and end times to closest five minute intervals
# so that we dont miss a candlestick by a few minutes and not wait more
# than 5 minutes to get it.
t_end = time.time()
t_end_min = t_end / 60
# floor it to the closest 5 minute mark. We add to the time to effectively widen our time period
t_end = (t_end_min + 5 - (t_end_min % 5)) * 60

t_start = t_end - LOOKBACK_SEC
t_start_min = t_start / 60
# we ceil it to the closes 5 minute mark. We subtract time to effectively get the previous closest
# 5 minute mark
t_start = t_start - (t_start_min % 5) * 60

P = poloniex(APIKey=config.api_key, Secret=config.secret_key)
resp = P.returnChartData(currencyPair="USDT_BTC", start=t_start, end=t_end, period=INTERVAL)
print resp
full_chart = {'{}'.format(datetime.datetime.fromtimestamp(int(r["date"])).strftime('%Y-%m-%d %H:%M:%S')): json.dumps(r) for r in resp}
batch_write_to_cache(full_chart)

while(True):
    logger.info("Start time: {} End time: {}".format(t_start, t_end))
    resp = P.returnChartData(currencyPair="USDT_BTC", start=t_start, end=t_end, period=INTERVAL)
    resp = {'{}'.format(datetime.datetime.fromtimestamp(int(r["date"])).strftime('%Y-%m-%d %H:%M:%S')): json.dumps(r) for r in resp}
    logger.info("RESP: {}".format(resp))
    batch_write_to_cache(resp)
    # @todo @logicissues. How to reduce latency in getting new candlesticks but by keeping the
    # pings low
    t_start = t_end
    time.sleep(INTERVAL - 120)
    # commenting this because going off of the latest time is more robust as it will
    # pick more data points if it slept for sometime
    #t_end = t_start + INTERVAL
    t_end = time.time()


