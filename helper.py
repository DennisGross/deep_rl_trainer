import argparse
import sys
from typing import Dict, Any

def get_arguments() -> Dict[str, Any]:
    """Parses all the command line arguments
    Returns:
        Dict[str, Any]: dictionary with the command line arguments as key and their assignment as value
    """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    arg_parser.add_argument('--checkpoint_path', help='Checkpoint Path', type=str,
                            default='')
    arg_parser.add_argument('--env', help='OpenAI Gym Environment', type=str,
                            default='CartPole-v1')
    arg_parser.add_argument('--rl', help='RL Algorithm', type=str, default='DQN')
    # training_iteration
    arg_parser.add_argument('--reward_threshold', help='Mean Reward that allows stop training', type=int, default=50)
    arg_parser.add_argument('--eval_interval', help='Number of environments per worker', type=int, default=10)
    arg_parser.add_argument('--num_workers', help='Number of Workers', type=int, default=4)
    arg_parser.add_argument('--num_envs_per_worker', help='Number of environments per worker', type=int, default=2)
    #arg_parser.add_argument('--interval', help='Interval of Ticker data', type=str, default='')

    args, _ = arg_parser.parse_known_args(sys.argv)
    return vars(args)

