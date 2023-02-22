"""Example data types functions which illustrate the core operations of our workflow"""
from pathlib import Path
from platform import node
from hashlib import sha512
from logging import getLogger

from redis import Redis

logger = getLogger(__name__)


def train_model(model_path: Path, database: Path, num_workers: int, redis_info: tuple[str, int]) -> tuple[Path, list[str]]:
    """Train a machine learning model cooperative with other workers

    Multiple instances of this function must be run across multiple workers.

    Args:
        model_path: Path to the current model
        database: Path to the training data
        num_workers: Number of cooperative workers to wait for
        redis_info: (Hostname, Port) for the redis server used to coordinate with other workers
    """

    # Get all cooperative hosts
    key = sha512(model_path).hexdigest()[:16]
    hosts = get_hosts(key, num_workers, redis_info)
    logger.info(f'Received list of {len(hosts)} hosts that will cooperate in model training')

    # Do the magic
    return model_path, hosts


def get_hosts(key: str, num_workers: int, redis_info: tuple[str, int] = ('localhost', 6379)) -> list[str]:
    """Get the host names of all other workers

    Args:
        key: Key used to identify this pool of workers
        num_workers: Number of cooperative workers to wait for
        redis_info: (Hostname, Port) for the redis server used to coordinate with other workers
    """
    # Make the redis client
    host, port = redis_info
    redis = Redis(host=host, port=port)

    # Immediately subscribe to the channel
    pubsub = redis.pubsub()
    channel = f'pubsub-{key}'
    pubsub.subscribe(channel)

    # Append my hostname to the list
    hostname = node()
    list_key = f'hosts-{key}'
    rank = redis.lpush(list_key, hostname)
    logger.info(f'I am rank {rank}')

    # Either wait for full list or wait
    if rank < num_workers:
        # Wait for someone else to publish the list
        hosts = None
        for message in pubsub.listen():
            if message['type'] == 'message':
                hosts = message['data'].decode().split(":")
                break
        pubsub.unsubscribe()
    elif rank == num_workers:
        # Send the lists of hosts to everyone else
        hosts = redis.lrange(list_key, 0, rank + 1)
        hosts = [h.decode() for h in hosts]

        redis.delete(list_key)  # No longer needed
        redis.publish(channel, ":".join(hosts))  # Everyone should be subscribed at this point
    else:
        raise ValueError(f'Received rank #{rank}, but there should only be {num_workers} ranks')

    return hosts
