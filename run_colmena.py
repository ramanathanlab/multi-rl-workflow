"""Implementation of the workflow in Colmena"""
from functools import partial, update_wrapper
from collections import defaultdict
from threading import Event, Condition
from pathlib import Path
from heapq import merge
from platform import node
import logging
import shutil
import sys

import numpy as np
from colmena.exceptions import TimeoutException
from colmena.models import Result
from colmena.task_server import ParslTaskServer
from colmena.thinker import BaseThinker, ResourceCounter, task_submitter, result_processor, event_responder
from colmena.queue.redis import RedisQueues

from multirl import rl, md, scoring
from multirl.models import Sequence
from multirl.parsl import build_polaris_single_job


def _wrap_function(func, *args, **kwargs):
    new_func = partial(func, *args, **kwargs)
    update_wrapper(new_func, func)
    return new_func


class Thinker(BaseThinker):
    """Implementation of the steering scheduler"""

    def __init__(
            self,
            queues: RedisQueues,
            output_dir: Path,
            starting_model: Path,
            nodes_per_training: int,
            training_slots: int,
            rollout_slots: int,
    ):
        """
        Args:
            queues: Queues used to communicate with Task Server
            output_dir: Directory in which to store output results
            starting_model: Path to the model to be trained and used to generate sequences
            nodes_per_training: Number of nodes to use to train policy
            training_slots: Number of execution slots set aside for training and simulation
            rollout_slots: Number of execution slots set aside for rollout and scoring sequences
        """
        super().__init__(queues, ResourceCounter(training_slots + rollout_slots, ["train", "rollout"]))
        self.nodes_per_training = nodes_per_training
        self.output_dir = output_dir
        self.scoring_functions = ['score_sequences']

        # Create the shared resources
        self.simulation_queue_lock: Condition = Condition()  # Protect the list from concurrent modification
        self.simulation_queue: list[tuple[float, Sequence]] = list()  # List of simulations to be executed
        self.database: dict[Sequence, dict[str, object]] = defaultdict(dict)  # Database of all sequences which have been evaluated
        self.current_model: Path = starting_model  # Path to the latest model
        self.seqs_being_scored: dict[str, dict[str, Sequence | int]] = {}  # Sequences currently being scored. Keyed on task_id of generation
        self.seqs_being_simulated: dict[str, Sequence] = {}  # Sequences being simulated

        # State associated with steering behavior
        self.start_training: Event = Event()  # Mark that we should start re-training the model
        self.training_nodes_gathered: int = 0  # Number of nodes currently set aside for training

        # Monitoring the progress of the application
        self.training_round: int = 0  # Current generation of the training round

        # Assign the workers to each pool
        self.rec.reallocate(None, "train", training_slots)
        self.rec.reallocate(None, "rollout", rollout_slots)

    def _save_result(self, result: Result, name: str):
        """Save a result to disk

        Args:
            result: Result to be saved
            name: Type of the result
        """
        with open(self.output_dir / f'{name}-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

    @task_submitter(task_type='train')
    def submit_md(self):
        """Submit a new molecular dynamics when resources are available"""

        # Get the next several task from the priority queue
        batch = []
        score = np.inf

        with self.simulation_queue_lock:
            # If there are not enough tasks, wait
            if len(self.simulation_queue) < 4:
                self.logger.info('There are insufficient simulations tasks available. Waiting...')
                self.simulation_queue_lock.wait()

            # Create my batch
            for _ in range(4):  # For jobs per batch
                score, seq = self.simulation_queue.pop()
                batch.append(seq)
        self.logger.info(f'Submitting next batch of {len(batch)} sequences. Minimum score: {score:.1e}. Queue length: {len(self.simulation_queue)}')

        # Submit task and store sequences in memory
        task_id = self.queues.send_inputs(
            batch,
            method='batch_run_molecular_dynamics',
            topic='simulate'
        )
        self.seqs_being_simulated[task_id] = batch

    @result_processor(topic='simulate')
    def store_md(self, result: Result):
        """Store the results from an MD computation in the database"""
        assert result.success, result.failure_info

        # Store our results in a database for later use
        my_seqs = self.seqs_being_simulated.pop(result.task_id)
        for seq, score in zip(my_seqs, result.value):
            self.database[seq]['md'] = score

        # Trigger retraining if enough data gathered
        if len(self.database) % 16 == 0:
            if self.start_training.is_set():
                self.logger.info('Enough data gathered for retraining, but training is still running')
            else:
                self.start_training.set()
                self.training_nodes_gathered = 0
                self.logger.info('Triggered retraining')
            self.start_training.set()

        # Set aside nodes for training by not marking them as free
        if self.start_training.is_set() and self.training_nodes_gathered < self.nodes_per_training:
            self.training_nodes_gathered += 1
            self.logger.info(f'Set aside {self.training_nodes_gathered} of {self.nodes_per_training} needed for training')
        else:
            self.rec.release('train')

        # Save the result
        self._save_result(result, 'simulation')

    @event_responder(event_name='start_training')
    def run_training(self):
        """Submits training tasks and collects results"""
        self.logger.info(f'Beginning training round {self.training_round}')

        # Start by submitting all required tasks, which will launch on nodes as needed
        for rank in range(self.nodes_per_training):
            self.queues.send_inputs(
                self.current_model,
                self.database,
                self.nodes_per_training,
                method='train_model',
                topic='train',
                task_info={'train_round': self.training_round, 'submitted_rank': rank}
            )
        self.logger.info(f'Submitted {self.nodes_per_training} training tasks')

        # Wait for all ranks to come back
        new_path: Path = self.current_model
        timeout: int | None = None
        for rank in range(self.nodes_per_training):
            try:
                result = self.queues.get_result(topic='train', timeout=timeout)
            except TimeoutException as err:
                raise ValueError('Training tasks did not exit properly.') from err
            self.logger.info(f'Collected training rank {rank + 1}/{self.nodes_per_training}')

            # Make sure it was successful
            assert result.success, result.failure_info

            # Release nodes back to simulation
            self.rec.release('train')

            # Save the result. Place the full host list in the task_info to save it
            new_path, my_rank, hosts = result.value
            result.task_info['rank'] = my_rank
            result.task_info['hosts'] = hosts
            self._save_result(result, 'training')

        # Update the current model
        self.current_model = new_path
        self.training_round += 1
        self.logger.info(f'Updated current model to {self.current_model}. Training round to {self.training_round}')
        self.done.set()

    @task_submitter(task_type='rollout')
    def submit_rollout(self):
        """Submit a rollout computation when nodes are free"""

        # Perform rollout with the latest model
        self.queues.send_inputs(
            self.current_model,
            4,  # Number of rollout episodes
            32,  # Number of sequences per batch
            method='policy_rollout',
            topic='rollout',
            task_info={'training_round': self.training_round}
        )

    @result_processor(topic='rollout')
    def store_rollout(self, result: Result):
        """Submit sequences generated by the rollout system to be scored"""
        # Make sure it worked
        assert result.success, result.failure_info

        # Collect the new sequences the result
        new_sequences = result.value
        result.task_info['num_generated'] = len(new_sequences)
        self.seqs_being_scored[result.task_id] = {
            'seqs': new_sequences,
            'num_done': 0
        }

        # Send them to the scoring function
        for screen_method in self.scoring_functions:
            self.queues.send_inputs(
                new_sequences,
                method=screen_method,
                topic='score',
                task_info={'num_to_screen': len(new_sequences), 'seqs_id': result.task_id, **result.task_info},
            )

        self._save_result(result, 'rollout')

    @result_processor(topic='score')
    def store_scores(self, result: Result):
        # Make sure it worked
        assert result.success, result.failure_info

        # Look up the sequences
        seqs_id = result.task_info['seqs_id']
        seq_info = self.seqs_being_scored[seqs_id]
        my_seqs = seq_info['seqs']

        # Add scores to the database
        for seq, score in zip(my_seqs, result.value):
            self.database[seq][result.method] = score

        # Check if we've completed all scoring functions of this sequences
        seq_info['num_done'] += 1
        all_done = seq_info['num_done'] == len(self.scoring_functions)
        self.logger.info(f'Finished {seq_info["num_done"]}/{len(self.scoring_functions)} scoring functions')
        if not all_done:
            return

        # If so, ...
        del self.seqs_being_scored[seqs_id]  # we no longer need to hold on to the sequences
        self.rec.release('rollout')  # and can start running another rollout

        # Compute a total score for every sequence in this batch
        def composite_score(known_scores: dict) -> float:
            """Get a total score given all data known about a certain sequence"""
            return sum(known_scores[m] for m in self.scoring_functions)

        total_score = [composite_score(self.database[seq]) for seq in my_seqs]

        # Then rank all sequences and add them to the simulation pool
        sorted_seqs = sorted(zip(total_score, my_seqs))
        with self.simulation_queue_lock:
            self.simulation_queue = list(merge(self.simulation_queue, sorted_seqs))

            # Unlock the simulation submitter
            self.simulation_queue_lock.notify_all()
            self.logger.info(f'Added {len(sorted_seqs)} to simulation queue. New length: {len(sorted_seqs)}')

        # Save result
        self._save_result(result, 'score')


if __name__ == "__main__":
    # Hard-coded arguments for now
    nodes_per_training = 2

    # Clear the test directory
    test_dir = Path('test-run')
    if test_dir.is_dir():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Set up the logging
    # Set up the logging
    handlers = [logging.FileHandler(test_dir / 'runtime.log'),
                logging.StreamHandler(sys.stdout)]


    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)


    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        handlers=handlers)

    # Make the queues used to coordinate between steering policy and task server
    queues = RedisQueues(
        topics=['train', 'rollout', 'reward', 'score', 'simulate'],
        keep_inputs=False,  # Do not send inputs back to the Thinker
        serialization_method='pickle',
    )

    # Pin arguments that do not change between invocations
    my_train_model = _wrap_function(rl.train_model, redis_info=(node(), 6379))

    # Make the Parsl configuration
    config = build_polaris_single_job(1, '/lus/grand/projects/CSC249ADCD08/multi-rl-workflow/env')

    # Make the task server
    doer = ParslTaskServer(
        methods=[
            (md.batch_run_molecular_dynamics, {'executors': ['training']}),
            (my_train_model, {'executors': ['training']}),
            (rl.policy_rollout, {'executors': ['rollout']}),
            (scoring.score_sequences, {'executors': ['rollout']})
        ],
        queues=queues,
        config=config,
    )
    doer.start()

    # Launch the thinker
    thinker = Thinker(
        queues=queues,
        output_dir=test_dir,
        starting_model=Path('not-real'),
        nodes_per_training=nodes_per_training,
        training_slots=2,
        rollout_slots=1
    )
    try:
        thinker.run()
    finally:
        # When done, kill the doer
        queues.send_kill_signal()
