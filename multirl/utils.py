import os
import logging
from platform import node
from typing import Callable

from redis import Redis

logger = logging.getLogger(__name__)


def pin_then_run(function: Callable, rank: int, n_ranks: int, *args, **kwargs):
    """Pin a function to certain CPU threads and a single GPU then run the function

    Args:
        function: Function to be run
        rank: Rank of the worker running this function
        n_ranks: Total number of workers
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
    """
    # Borrowing some logic from Parsl for pinning: https://github.com/Parsl/parsl/blob/master/parsl/executors/high_throughput/process_worker_pool.py#L542
    # Pick a block of cores for this worker
    avail_cores = sorted(os.sched_getaffinity(0))  # Get the available processors
    cores_per_worker = len(avail_cores) // n_ranks
    assert cores_per_worker > 0, "Affinity does not work if there are more workers than cores"

    my_cores = avail_cores[cores_per_worker * rank:cores_per_worker * (rank + 1)]
    os.sched_setaffinity(0, my_cores)

    # Pin to a single GPU
    assert "CUDA_VISIBLE_DEVICES" not in os.environ, "CUDA_VISIBLE_DEVICES is already set"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # Run the function
    try:
        return function(*args, **kwargs)
    finally:
        # Unset affinity
        os.sched_setaffinity(0, avail_cores)
        os.environ.pop("CUDA_VISIBLE_DEVICES")


def get_hosts(key: str, num_workers: int, redis_info: tuple[str, int] = ('localhost', 6379)) -> tuple[int, list[str]]:
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
    rank = redis.lpush(list_key, hostname) - 1  # Change workers to be 0-based numbering
    logger.info(f'I am rank {rank} for {key}')

    # Either wait for full list or wait
    if rank < num_workers - 1:
        # Wait for someone else to publish the list
        hosts = None
        for message in pubsub.listen():
            if message['type'] == 'message':
                hosts = message['data'].decode().split(":")
                break
    elif rank == num_workers - 1:
        # Send the lists of hosts to everyone else
        hosts = redis.lrange(list_key, 0, rank + 1)
        hosts = [h.decode() for h in hosts]
        assert len(hosts) == num_workers, f'We did not find the correct number of workers. Expected: {num_workers}. Found: {len(hosts)}.'

        redis.delete(list_key)  # No longer needed
        num_received = redis.publish(channel, ":".join(hosts))  # Everyone should be subscribed at this point
        assert num_received == num_workers, f'Some workers did not receive host list. Expected: {num_workers}. Found: {num_received}'
    else:  # pragma: no cover
        raise ValueError(f'Received rank #{rank}, but there should only be {num_workers} ranks')

    # Close the pubsub now that we're done
    pubsub.unsubscribe()
    pubsub.close()

    return rank, hosts
