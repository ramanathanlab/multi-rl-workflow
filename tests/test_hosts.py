from concurrent.futures import ThreadPoolExecutor

from redis.client import Redis

from multirl.utils import get_hosts


def test_hosts():
    # Ensure we start with a blank slate
    redis = Redis()
    redis.delete('hosts-test')

    # Try with a single hosts
    my_rank, hosts = get_hosts('test', 1)
    assert my_rank == 0
    assert len(hosts) == 1

    # Try with 4 hosts (anything more than 1 should test both code paths)
    with ThreadPoolExecutor() as thr:
        futures = [thr.submit(get_hosts, 'test', 4) for _ in range(4)]

        # Get the hosts of the first one
        my_rank, hosts = futures[0].result(timeout=4)
        assert my_rank < 4
        assert len(hosts) == 4

        # Make sure all yield the same result
        observed_ranks = set()
        for future in futures:
            my_rank, my_hosts = future.result(timeout=1)
            observed_ranks.add(my_rank)
            assert my_hosts == hosts
        assert observed_ranks == {0, 1, 2, 3}
