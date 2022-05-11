from helper import *
import ray.rllib.agents.ppo as ppo




if __name__ == "__main__":
    command_line_arguments = get_arguments()
    env_class, env = create_env(command_line_arguments['env'], command_line_arguments['env_config'])
    
    checkpoint_path = get_checkpoint_path(command_line_arguments['run_id'], command_line_arguments['tracking_uri'])
    config={
            'model': {
                'fcnet_hiddens' : [100, 100],
                'fcnet_activation' : "tanh"
            },
            "env_config": parse_env_config(command_line_arguments['env_config'])
        }
    agent = ppo.PPOTrainer(config=config, env=env_class)
    agent.restore(checkpoint_path)
    episode_reward = 0
    done = False
    for epoch in range(0,command_line_arguments['eval_interval']):
        obs = env.reset()
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        print(f"{epoch+1}.Episode Reward: {episode_reward}")

    clean_up()