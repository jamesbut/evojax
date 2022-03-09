import argparse
import os

from evojax.task.bipedal_walker import BipedalWalker
from evojax.policy import MLPPolicy
from evojax.algo import CMA
from evojax import util
from evojax import Trainer


def main(config):

    log_dir = './log/bipedal_walker'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='CartPole', log_dir=log_dir, debug=config.debug)

    logger.info('EvoJAX BipedalWalker Demo')
    logger.info('=' * 30)

    task = BipedalWalker()

    policy = MLPPolicy(
        input_dim=task.obs_shape[0],
        hidden_dims=[],
        output_dim=task.act_shape[0]
    )

    solver = CMA(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        init_stdev=config.init_std,
        seed=config.seed,
        logger=logger,
    )

    # Train
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=task,
        test_task=task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=100, help='NE population size.')
    parser.add_argument(
        '--init-std', type=float, default=1.0, help='Initial std.')
    parser.add_argument(
        '--seed', type=int, default=108, help='Random seed for training.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    parser.add_argument(
        '--max-iter', type=int, default=100, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=100, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=20, help='Logging interval.')

    config, _ = parser.parse_known_args()
    return config


if __name__ == '__main__':
    config = parse_args()
    main(config)
