import os

# poloniex keys
api_key = os.getenv('APIKEY', '')
secret_key = os.getenv('SECRETKEY', '')


# redis where video ids are cached.
redis_ip = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", 6379)
redis_db = os.getenv("REDIS_DB", 0)
redis_ttl = os.getenv("REDIS_TTL", 60 * 60 * 24 * 30 * 6)

# orderbook data
redis_ob_db = os.getenv("REDIS_OB_DB", 0)
redis_ob_ttl = os.getenv("REDIS_TTL", 60 * 60 * 24 * 14)
data_dir = os.getenv("DATA_DIR", "/Users/tstanley/Documents/repos/side_projects/crypto/data/")