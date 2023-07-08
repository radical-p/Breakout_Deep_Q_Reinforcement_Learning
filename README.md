# Deep Q-Network (DQN) for Breakout Game

This project implements the Deep Q-Network (DQN) algorithm to train an agent to play the Breakout game from the Atari environment using a reinforcement learning approach.

## Requirements

To run this project, you need to have the following libraries installed:

- NumPy
- Pandas
- Gym
- Keras

## Usage

1. Clone the repository to your local machine.
2. Make sure you have the required dependencies installed.
3. Open the script file `dqn_breakout.py`.
4. Adjust the training parameters if needed:
   - `episodes`: The number of episodes to train the agent (default: 100,000).
   - `trial_len`: The maximum number of steps per episode (default: 10,000).
5. Save the script file.
6. Run the script using a Python interpreter: `python dqn_breakout.py`.
7. The training process will start, and the agent will learn to play the Breakout game.
8. During training, the average reward per 100 episodes will be recorded and displayed using Matplotlib.
9. Once the training is complete, the script will output the reward and episode information.
10. You can visualize the training progress by examining the reward and episode lists.
