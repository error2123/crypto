from api import *
import json
import logging
import sys
import config
import datetime
from redis_client import batch_write_to_cache

# make sure the end period is a month higher. so we circumvent with
# bug I commented on above the "While" loop
#python populator_backfill.py "2017-01-01" "2017-05-01"


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

INTERVAL = 60 * 5
pair = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]
P = poloniex(APIKey=config.api_key, Secret=config.secret_key)
#import ipdb; ipdb.set_trace()
start_ut = time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple())
end_ut = time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple())

tmp_start = start_ut
tmp_end = start_ut + (60 * 60 * 24 * 30)
#@todo the while loop misses on the last pass @FIXIT
while(tmp_end < end_ut):
    logger.info("Start time: {} End time: {}".format(datetime.datetime.fromtimestamp(tmp_start),
                                                     datetime.datetime.fromtimestamp(tmp_end)))
    resp = P.returnChartData(currencyPair=pair, start=tmp_start, end=tmp_end, period=INTERVAL)
    resp = {'{}###{}'.format(pair, datetime.datetime.fromtimestamp(int(r["date"])).strftime('%Y-%m-%d %H:%M:%S')): json.dumps(r)
            for r in resp}
    logger.info("RESP: {}".format(resp))
    batch_write_to_cache(resp)
    tmp_start = tmp_end
    tmp_end = tmp_start + (60 * 60 * 24 * 30)
    time.sleep(int(sys.argv[4]))
    
