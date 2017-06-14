from api import *
import json
import logging
import sys
import config
import datetime
from redis_client import batch_write_to_cache

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

P = poloniex(APIKey=config.api_key, Secret=config.secret_key)

while(True):
    time_t = time.time()
    resp = P.returnOrderBook(currencyPair="all")
    print "TIMESTAMP:", time_t, datetime.datetime.fromtimestamp(int(time_t)).strftime('%Y-%m-%d %H:%M:%S')
    resp = {'order_book###{}'.format(datetime.datetime.fromtimestamp(int(time_t)).strftime('%Y-%m-%d %H:%M:%S')): json.dumps(resp)}
    logger.info("RESP: {}".format(resp))
    batch_write_to_cache(resp)
    time.sleep(30)


