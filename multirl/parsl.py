"""Utilities for making Parsl configurations"""
import logging
import os
from pathlib import Path

from parsl import Config, HighThroughputExecutor
from parsl.launchers import WrappedLauncher
from parsl.providers import LocalProvider

logger = logging.getLogger(__name__)


def build_polaris_single_job(rollout_size: int | float, conda_path: str) -> Config:
    """Assemble a Polaris configuration where all workers are part of the same job.

    The configuration assumes that Parsl is being run inside the job and can access
    the HOSTFILE via environment variables. It will launch workers for training/simulation
    on one set of nodes and partition off a second set to use for rollout and scoring.

    Args:
        rollout_size: Either the number of nodes (if >=1) or fraction of nodes (if <1)
            being used for rollout and scoring
        conda_path: Path to the Anaconda
    Returns:
        Parsl configuration
    """

    # Get the HOSTFILE from the environment variables
    assert 'PBS_NODEFILE' in os.environ, 'Cannot find hostfile'
    node_file = os.environ['PBS_NODEFILE']
    with open(node_file) as fp:
        nodes = [x.strip() for x in fp]
    logger.info(f'Found {len(nodes)} in {node_file}')

    # Split them based on the
    total_nodes = len(nodes)
    if total_nodes < 2:
        raise ValueError('You must allocate at least two nodes')
    if rollout_size < 1:
        num_rollout = rollout_size * total_nodes
    else:
        num_rollout = rollout_size
    num_rollout = max(1, num_rollout)
    logger.info(f'Allocated {num_rollout} nodes to rollout. Will use {total_nodes - num_rollout} for training')

    # Split the host file into separate nodes
    job_name = os.environ.get('PBS_JOBID', 'jobname')
    training_nodefile = Path(f'hostfile.training.{job_name}')
    rollout_nodefile = Path(f'hostfile.rollout.{job_name}')
    training_nodefile.write_text('\n'.join(nodes[num_rollout:]))
    rollout_nodefile.write_text('\n'.join(nodes[:num_rollout]))

    # Assemble the config
    worker_init = f'''module load conda
conda activate {conda_path}
which python
hostname
'''
    return Config(
        executors=[
            HighThroughputExecutor(
                label='training',
                max_workers=1,  # One worker will use the whole node
                provider=LocalProvider(
                    min_blocks=1,
                    max_blocks=1,
                    worker_init=worker_init,
                    launcher=WrappedLauncher(
                        f"mpiexec -n {total_nodes - num_rollout} --ppn 1 --hostfile {training_nodefile} --depth=64 --cpu-bind depth"
                    )
                )
            ),
            HighThroughputExecutor(
                label='rollout',
                max_workers=1,  # One worker will use the whole node
                provider=LocalProvider(
                    min_blocks=1,
                    max_blocks=1,
                    worker_init=worker_init,
                    launcher=WrappedLauncher(
                        f"mpiexec -n {num_rollout} --ppn 1 --hostfile {rollout_nodefile} --depth=64 --cpu-bind depth"
                    )
                )
            )
        ]
    )


def build_local_configuration(n_gpus: int, rollout_size: int | float):
    """Assemble a local configuration with a single GPU for each task.

    Args:
        n_gpus: Number of GPUs available for workers
        rollout_size: Number or fraction of GPUs to use for rollout
    Returns:
        Parsl configuration
    """

    # Partition the GPUs
    gpus = list(range(n_gpus))
    if n_gpus < 2:
        raise ValueError('You must have at least two GPUs')
    if rollout_size < 1:
        num_rollout = rollout_size * n_gpus
    else:
        num_rollout = rollout_size
    num_rollout = max(1, num_rollout)
    logger.info(f'Allocated {num_rollout} GPUs to rollout. Will use {n_gpus - num_rollout} for training')

    rollout_gpus = [str(x) for x in gpus[:num_rollout]]
    train_gpus = [str(x) for x in gpus[num_rollout:]]

    return Config(
        executors=[
            HighThroughputExecutor(
                label='training',
                available_accelerators=train_gpus,
                provider=LocalProvider(
                    min_blocks=1,
                    max_blocks=1,
                )
            ),
            HighThroughputExecutor(
                label='rollout',
                available_accelerators=rollout_gpus,
                provider=LocalProvider(
                    min_blocks=1,
                    max_blocks=1,
                )
            )
        ]
    )
