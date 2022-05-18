from helper import *
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from typing import Dict
import os
from numpy import savetxt
import numpy as np
import mlflow
import os
import json

def prepare_config(command_line_arguments: Dict) -> Dict:
    """Prepare config from command line arguments

    Args:
        command_line_arguments (dict): Command Line Arguments

    Returns:
        dict: Config
    """
    mlflow.set_tracking_uri(command_line_arguments['tracking_uri'])
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))
    print(mlflow.get_run(command_line_arguments['run_id']).to_dictionary())
    model = json.loads(mlflow.get_run(command_line_arguments['run_id']).to_dictionary()['data']['params']['model'].replace("'",'"'))
    config={
            'model':model,
            "env_config": parse_env_config(command_line_arguments['env_config'])
        }
    return config


if __name__ == "__main__":
    FILE_LENGTH = 5
    command_line_arguments = get_arguments()
    env_class, env = create_env(command_line_arguments['env'], command_line_arguments['env_config'])
    #client = MlflowClient(tracking_uri=command_line_arguments['tracking_uri'])
    config = prepare_config(command_line_arguments)
    checkpoint_path = get_checkpoint_path(command_line_arguments['run_id'], command_line_arguments['tracking_uri'])
    agent = None
    rl = parse_rl(command_line_arguments)
    if rl.lower() == "dqn":
        agent = dqn.DQNTrainer(config=config, env=env_class)
    elif rl.lower() == "ppo":
        agent = ppo.PPOTrainer(config=config, env=env_class)
    else:
        NotImplementedError("RL Agent is not supported")
    agent.restore(checkpoint_path)
    for epoch in range(0,command_line_arguments['eval_interval']):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.compute_action(obs)
            obs_old = np.array(obs, copy=True)  
            obs, reward, done, info = env.step(action)
            if command_line_arguments['state_collection_path']!="/":
                obs_old_filepath, obs_old_filename = get_new_random_filename(command_line_arguments['state_collection_path'], FILE_LENGTH)
                obs_filepath, obs_filename = get_new_random_filename(command_line_arguments['state_collection_path'], FILE_LENGTH)
                obs_old_filepath = obs_old_filepath+ "_"+ str(action)+"_"+str(reward) + "_" + str(obs_filename)
                savetxt(obs_old_filepath, obs, delimiter=',')
                savetxt(obs_filepath, obs, delimiter=',')
                
            episode_reward += reward
        print(f"{epoch+1}.Episode Reward: {episode_reward}")


    clean_up()