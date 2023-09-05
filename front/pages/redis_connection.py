import redis

class RedisConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisConnection, cls).__new__(cls)
            cls._instance.init_connection()
        return cls._instance

    def init_connection(self):
        # Connect to Redis server
        self.connection = redis.Redis(host='localhost', port=6379, db=0)
        print("Connected to Redis server.")


    def get_connection(cls):
        return cls.connection