import os
from pathlib import Path

from pytest import raises

from multirl.parsl import build_polaris_single_job, build_local_configuration


def test_polaris_single(tmp_path):
    os.environ['PBS_NODEFILE'] = 'hostfile'

    def make_hostfile(n):
        with open('hostfile', 'w') as fp:
            for i in range(n):
                print(f'host.{i}', file=fp)

    # Ensure we crash if only one node
    with raises(ValueError) as error:
        make_hostfile(1)
        build_polaris_single_job(0.1, '/some/path')
    assert 'two nodes' in str(error)

    # Test with fractions
    make_hostfile(2)
    build_polaris_single_job(0.1, '/some/path')
    assert Path('hostfile.rollout.jobname').read_text() == 'host.0'
    assert Path('hostfile.training.jobname').read_text() == 'host.1'

    # Test with total numbers
    make_hostfile(2)
    build_polaris_single_job(1, '/some/path')
    assert Path('hostfile.rollout.jobname').read_text() == 'host.0'
    assert Path('hostfile.training.jobname').read_text() == 'host.1'


def test_local_gpus(tmp_path):
    # Test a configuration with total node count
    config = build_local_configuration(4, 1)
    assert config.executors[0].label == 'training'
    assert config.executors[0].available_accelerators == ['1', '2', '3']
    assert config.executors[1].available_accelerators == ['0']

    # Test a configuration with a minimum fraction of nodes
    config = build_local_configuration(4, 0.1)
    assert config.executors[0].label == 'training'
    assert config.executors[0].available_accelerators == ['1', '2', '3']
    assert config.executors[1].available_accelerators == ['0']

    # Ensure it balks if you have fewer than two GPUs
    with raises(ValueError) as error:
        build_local_configuration(1, 1)
    assert 'two GPUs' in str(error.value)
