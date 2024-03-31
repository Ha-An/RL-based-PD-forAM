from GYM_wrapper import GymInterface
from config import *
import numpy as np
import time
import os
from stable_baselines3 import PPO  # DDPG

# Create environment
env = GymInterface()
# env.clean_up_logs() ###
# result_folder = "../results"###
# Define a function to evaluate a trained model
import shutil

def delete_files_in_directory(directory):
    """디렉토리 내의 모든 파일을 삭제합니다."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"파일 삭제 중 오류 발생: {e}")

# 결과 디렉토리와 로그 디렉토리를 지정합니다.
RESULTS_DIR = "../results"
LOGS_DIR = "../logs"

# 결과 디렉토리와 로그 디렉토리 내의 파일을 삭제합니다.
delete_files_in_directory(RESULTS_DIR)
delete_files_in_directory(LOGS_DIR)

def evaluate_model(model, env, num_episodes, result_folder):
    #    env = GymInterface()
    all_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        all_rewards.append(episode_reward)
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    os.makedirs(result_folder, exist_ok=True)

    for x in range(len(env.decomposed_parts)):
        mesh = env.PD_tree[env.decomposed_parts[x]]["Mesh"]
        mesh.export(os.path.join(result_folder, f"{env.decomposed_parts[x]}.stl")) ###
    return mean_reward, std_reward


# Train the agent
start_time = time.time()
model = PPO("MlpPolicy", env, verbose=0, device='cuda')
model.learn(total_timesteps=N_EPISODES)  # Time steps = episodes in our case
model.save("./Saved_RL_Models/PPO")
env.render()  # Render the environment to see how well it performs


# Evaluate the trained agent
mean_reward, std_reward = evaluate_model(model, env, N_EVAL_EPISODES, result_folder)
print(
    f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
end_time = time.time()
print(f"Computation time: {(end_time - start_time)/3600:.2f} hours")


# TensorBoard 실행:
# tensorboard --logdir="C:/tensorboard_logs/"
# http://localhost:6006/
