import argparse
import sys
from typing import Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
import glob
import os
import shutil
import importlib
import gym
import random
import string



TMP_DIR = './tmp'
TUNE_DIR = 'tune_dir'

def get_arguments() -> Dict[str, Any]:
    """
    Parses all the command line arguments
    Returns:
        Dict[str, Any]: dictionary with the command line arguments as key and their assignment as value
    """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()
    arg_parser.add_argument('--run_id', help='Checkpoint Path', type=str,
                            default='')
    arg_parser.add_argument('--env', help='OpenAI Gym Environment', type=str,
                            default='')
    arg_parser.add_argument('--env_config', help='Environment Config', type=str,
                            default='')
    arg_parser.add_argument('--rl', help='RL Algorithm', type=str, default='DQN')

    # Neural Network
    arg_parser.add_argument('--fcnet_activation', help='Activation Function', type=str, default='tanh')
    arg_parser.add_argument('--fcnet_hiddens', help='Hidden Layer Neurons', type=str, default='100,100')
    # training_iteration
    arg_parser.add_argument('--reward_threshold', help='Mean Reward that allows stop training', type=int, default=50)
    arg_parser.add_argument('--eval_interval', help='Number of environments per worker', type=int, default=10)
    # Technical
    arg_parser.add_argument('--tracking_uri', help='The tracking URI for where to manage experiments and runs. This can either be a local file path or a remote server. ', type=str, default='')
    arg_parser.add_argument('--ray_head', help='', type=str, default='')
    
    arg_parser.add_argument('--num_workers', help='Number of Workers', type=int, default=4)
    arg_parser.add_argument('--num_envs_per_worker', help='Number of environments per worker', type=int, default=2)
    arg_parser.add_argument('--state_collection_path', help='Path to the state collection folder', type=str, default='')

    args, _ = arg_parser.parse_known_args(sys.argv)
    return vars(args)

def get_checkpoint_path(run_id, tracking_uri):
    checkpoint_path = ''
    if run_id!="":
        client = MlflowClient(tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        if os.path.exists(TMP_DIR) == False:
            os.mkdir(TMP_DIR)
        client._tracking_client.download_artifacts(run_id, '.', TMP_DIR)
        all_sub_paths = glob.glob(TMP_DIR + "/*")
        for path in all_sub_paths:
            if path.startswith(TMP_DIR + "/checkpoint_"):
                potential_checkpoints = glob.glob(path+"/*")
                for check in potential_checkpoints:
                    head, tail = os.path.split(check)
                    if tail.startswith("checkpoint-") and tail.find('.')==-1:
                        checkpoint_path = check
                        
                break
    print("Checkpoint-Path:", checkpoint_path)
    return checkpoint_path

def create_env(env_id, env_config):
    try:
        return env_id, gym.make(env_id)
    except Exception as msg:
        print(msg)
        env_module = importlib.import_module(env_id)
        return env_module.MyEnvironment, env_module.MyEnvironment(env_config)

def parse_env_config(env_config):
    config = {}
    if str(env_config).strip()=="":
        return config
    for part in env_config.split(";"):
        key, value = part.split(":")
        config[key] = value
    return config

def parse_hidden_layer_neurons(fcnet_hiddens):
    hiddens = []
    for hidden in fcnet_hiddens.split(","):
        hiddens.append(int(hidden))
    return hiddens

def get_new_random_filename(state_collection_path, length):
    filename = get_random_string(length)
    filepath = os.path.join(state_collection_path, filename)
    while os.path.isfile(filepath):
        filename = get_random_string(length)
        filepath = os.path.join(state_collection_path, filename)
    return filepath, filename


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def clean_up():
    try:
        shutil.rmtree(TMP_DIR, ignore_errors=False, onerror=None)
    except:
        pass
    try:
        shutil.rmtree(TUNE_DIR, ignore_errors=False, onerror=None)
    except:
        pass