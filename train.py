from click import command
import ray
from ray import tune
from helper import *
import importlib
from ray.tune.integration.mlflow import MLflowLoggerCallback


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
            "env_config": parse_env_config(command_line_arguments['env_config'])
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    config["env"], _ = create_env(command_line_arguments['env'], command_line_arguments['env_config'])

    analysis = tune.run(
        command_line_arguments['rl'],
        local_dir=TUNE_DIR,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        restore=get_checkpoint_path(command_line_arguments['run_id'], command_line_arguments['tracking_uri']),
        stop={'episode_reward_mean': command_line_arguments['reward_threshold']},
        config=config,
        callbacks=[MLflowLoggerCallback(save_artifact=True, tracking_uri=command_line_arguments['tracking_uri'])]
    )
    clean_up()
