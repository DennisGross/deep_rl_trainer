import ray
from ray import tune
from helper import *
import importlib
import mlflow

if __name__ == "__main__":
    command_line_arguments = get_arguments()

    ray.init()
    config={
            "evaluation_interval": command_line_arguments['eval_interval'],
            "num_workers": command_line_arguments['num_workers'],
            "num_envs_per_worker": command_line_arguments['num_envs_per_worker'],
            'model': {
                'fcnet_hiddens' : [100, 100],
                'fcnet_activation' : "tanh"
            },
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if command_line_arguments['env']=="env":
        env_module = importlib.import_module(command_line_arguments['env'])
        config["env"] = env_module.MyEnvironment
    else:
        config["env"] = command_line_arguments['env']
    if command_line_arguments['checkpoint_path']=="/":
        command_line_arguments['checkpoint_path'] = ""

    analysis = tune.run(
        command_line_arguments['rl'],
        local_dir="../tune_dir",
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        #metric="episode_reward_mean",
        restore=command_line_arguments['checkpoint_path'],
        stop={'episode_reward_mean': command_line_arguments['reward_threshold']},
        config=config
    )
    trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max")
    print("Checkpoint-Path:", trial.checkpoint)