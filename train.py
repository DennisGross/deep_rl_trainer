import ray
from ray import tune
from helper import *
from ray.tune.integration.mlflow import MLflowLoggerCallback


if __name__ == "__main__":
    command_line_arguments = get_arguments()
    if command_line_arguments['ray_head']!="":
        ray.init(address=command_line_arguments['ray_head'])
    else:
        ray.init()
    config={
            "evaluation_interval": command_line_arguments['eval_interval'],
            "num_workers": command_line_arguments['num_workers'],
            "num_envs_per_worker": command_line_arguments['num_envs_per_worker'],
            #"num_gpus": 0,
            #"num_cpus": 1,
            'model': {
                'fcnet_hiddens' : parse_hidden_layer_neurons(command_line_arguments),
                'fcnet_activation' : parse_activation_function(command_line_arguments)
            },
            
            "env_config": parse_env_config(command_line_arguments['env_config'])
            #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    config["env"], _ = create_env(command_line_arguments['env'], command_line_arguments['env_config'])
    print(command_line_arguments)
    analysis = tune.run(
        command_line_arguments['rl'],
        local_dir=TUNE_DIR,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        restore=get_checkpoint_path(command_line_arguments['run_id'], command_line_arguments['tracking_uri']),
        stop={'episode_reward_mean': command_line_arguments['reward_threshold']},
        config=config,
        callbacks=[MLflowLoggerCallback(save_artifact=True, experiment_name='Default', tracking_uri=command_line_arguments['tracking_uri'])]
    )
    clean_up()
