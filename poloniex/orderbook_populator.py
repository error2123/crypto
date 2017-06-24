from api import *
import json
import logging
import sys
import config
import datetime
import os

from redis_client import batch_write_to_cache, redis_cli_ob

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
P = poloniex(APIKey=config.api_key, Secret=config.secret_key)
#pair = "USDT_ETH"
pair = sys.argv[1]
sleep_t = 5
inter_sleep = 1
prev_t = time.time()
time.sleep(sleep_t)

while(True):
    time_t = time.time()
    try:
        resp = P.returnOrderBook(currencyPair=pair)
        print "TIMESTAMP:", time_t, datetime.datetime.fromtimestamp(int(time_t)).strftime('%Y-%m-%d %H:%M:%S')
        day = str(datetime.datetime.fromtimestamp(int(time_t)).strftime('%Y-%m-%d'))
        dtime = str(datetime.datetime.fromtimestamp(int(time_t)).strftime('%Y-%m-%d-%H-%M-%S'))
        dtime_prev = str(datetime.datetime.fromtimestamp(int(prev_t)).strftime('%Y-%m-%d-%H-%M-%S'))
        #resp = {'order_book###{}'.format(datetime.datetime.fromtimestamp(int(time_t)).strftime('%Y-%m-%d %H:%M:%S')): json.dumps(resp)}
        #logger.info("RESP: {}".format(resp))
        resp = json.dumps(resp)
        dir_path = config.data_dir + "/" + pair + "/" + day
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(dir_path + "/" + dtime + ".depth", 'w') as f:
            f.write(resp)
        time.sleep(inter_sleep)
        resp_oh = P.returnMarketTradeHistory(currencyPair=pair, start=prev_t, end=time_t)
        print resp_oh, prev_t, time_t
        resp_oh = json.dumps(resp_oh)
        with open(dir_path + "/" + dtime_prev + "_" + dtime + ".trades", 'w') as f:
            f.write(resp_oh)
        #batch_write_to_cache(resp, ttl=config.redis_ob_ttl, client=redis_cli_ob)

    except Exception, e:
        print e
    prev_t = time_t
    time.sleep(sleep_t-inter_sleep)
    

