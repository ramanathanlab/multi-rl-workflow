"""Test the placeholder functions"""
from pathlib import Path
from time import perf_counter

from multirl.models import Sequence
from multirl.scoring import score_sequences
from multirl.rl import train_model, policy_rollout
from multirl.md import batch_run_molecular_dynamics


def test_train():
    # Make sure it gets host names
    new_path, my_rank, hosts = train_model(Path('model-path'), Path('data-path'), num_workers=1, redis_info=('localhost', 6379))
    assert my_rank == 0
    assert len(hosts) == 1


def test_rollout():
    # Make sure it caches the model
    start_time = perf_counter()
    policy_rollout(Path('model-path'), 2, batch_size=8)
    assert perf_counter() - start_time > 0.5

    start_time = perf_counter()
    new_seqs = policy_rollout(Path('model-path'), 2, 2)
    assert len(new_seqs) == 4
    assert perf_counter() - start_time < 0.5


def test_score():
    scores = score_sequences([Sequence(), Sequence()])
    assert len(scores) == 2


def test_batch_md():
    results = batch_run_molecular_dynamics([Sequence(), Sequence()])
    # Results should be the name of the CUDA device
    assert set(results) == {"0", "1"}
