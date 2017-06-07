import redis
import json
import logging.config

import config
logger = logging.getLogger(__name__)

redis_cli = redis.StrictRedis(host=config.redis_ip,
                              port=config.redis_port,
                              db=config.redis_db)

def put_in_cache(hash_key, output, ttl=config.redis_ttl):
    """
    cahe the video object
    :param hash_key: doc id from solr
    :param output: serialized video object
    :param ttl: time to live
    :return:
    """
    logger.debug("Attempting to write key-value {} {} to redis".format(hash_key, output))
    return redis_cli.set(hash_key, output, ex=ttl)

def get_from_cache(hash_key):
    """
    get the serialized video object from redis
    :param hash_key: solr doc id
    :return: serialized video object
    """
    logger.debug("Attempting to get key {} from redis".format(hash_key))
    return redis_cli.get(hash_key)

#@performanceimprovement @todo, the get key by regex are expensive as they
# search all key patterns
def get_keys_by_pattern(pattern):
    """
    get the serialized video object from redis
    :param hash_key: solr doc id
    :return: serialized video object
    """
    logger.debug("Attempting to get all keys with pattern {} from redis".format(pattern))
    return redis_cli.keys(pattern=pattern)


#@todo all these batch functions can be made more efficient @performanceimprovement
# make sure all json calls are  using cjson as well
def batch_write_to_cache(objs, ttl=config.redis_ttl):
    """
    cahe the video object
    :param hash_key: doc id from solr
    :param output: serialized video object
    :param ttl: time to live
    :return:
    """
    for x in objs:
        logger.debug("Attempting to write key-value {} {} to redis".format(x, objs[x]))
        put_in_cache(x, objs[x])



def batch_read_from_cache(dates):
    """
    Tries to retrieve a batch of keys and returns a tuple  of video
    objects and a list of uncached keys
    :param ids:
    :return:
    """
    objs = {}
    for date in dates:
        # @todo is there a interface to read all keys in one go?
        keys = get_keys_by_pattern('{}*'.format(date))
        for key in keys:
            obj = get_from_cache(key)
            objs[key] = json.loads(obj)
    return objs

