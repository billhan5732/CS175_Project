---
layout: default
title: Status
---

## Project Summary

 This project is about training an AI agent to control a Minecraft player in a boat to race around a race track made of ice. The ice makes the track have low friction, making it difficult to master movement mechanics. We are training it using general Reinforcement Learning techniques
 and experimenting with various models.

https://youtu.be/qui1wv2CaBU

[![Video Title](https://img.youtube.com/vi/qui1wv2CaBU/0.jpg)](https://youtu.be/qui1wv2CaBU)

## Approach

Our agent is trained to autonomously navigate a Minecraft boat along procedurally generated ice tracks using reinforcement learning. The environment is implemented using Project Malmo and exposed as a custom OpenAI Gym environment. The agent learns to drive a boat from one checkpoint to the next while avoiding hazards such as lava and minimizing inefficient behaviors such as spinning or getting stuck.

We use the Soft Actor-Critic (SAC) algorithm (Haarnoja et al., 2018), implemented using the Stable-Baselines3 library. SAC is an off-policy actor-critic algorithm designed for continuous control tasks. It learns a stochastic policy 
that maximizes both expected reward and policy entropy. The objective is:

$$
J(\pi) =
\mathbb{E}_{(s_t,a_t) \sim \rho_{\pi}}
\left[
\sum_{t} r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))
\right]
$$

We use the default multilayer perceptron policy (MlpPolicy) from Stable-Baselines3.
  
The Action space is:  
Action	Range	Description  
Throttle	[-1, 1]	Forward or backward movement  
Steering	[-1, 1]	Left or right turning  

Actions are converted into discrete Minecraft control commands (forward, back, left, right) using a threshold of 0.1.


## Evaluation

Early versions of the boat-racing agent had several limitations that reduced performance and stability. Initial experiments used PPO, but training was slow due to the cost of Minecraft simulation and the agent often converged to unstable behaviors such as spinning in place. The original observation space also lacked sufficient rotational information, making it difficult for the agent to align itself with the next checkpoint.

The current implementation improved performance through several changes. We switched from PPO to Soft Actor-Critic (SAC) to improve sample efficiency and control granularity. Rotational information was added to the observation space using the cosine and sine of the angle to the next checkpoint, and distance-based reward shaping was introduced to encourage steady progress toward the target. Additional penalties were added to discourage spinning and getting stuck.

## Remaining Goals and Challenges

Currently, the agent is being trained in small, incremental steps: it is spawned at a checkpoint and trained to reach only the next checkpoint. While this approach allows the agent to learn basic navigation skills in a controlled setting, it is still far from our final goal of completing entire tracks autonomously. At present, the agent successfully reaches the target checkpoint roughly one-third of the time, indicating that additional training and reward refinement are necessary to improve consistency and performance.

Our immediate goals are to continue incremental training while refining the reward function to better encourage forward progress, reduce spinning, and maintain alignment with the track. Once the agent reliably reaches individual checkpoints, we aim to extend training to full-track navigation, requiring it to string together multiple checkpoint sequences. Challenges we anticipate include maintaining stability over longer episodes, handling diverse track layouts, and ensuring the agent generalizes across tracks rather than overfitting to specific spawn points. To address these issues, we plan to experiment with reward shaping, further enrich the observation space, and potentially compare with alternative RL algorithms or hyperparameter settings to improve sample efficiency and learning stability.

## Resources Used

- Malmo Documentation
- Project Malmo Github Repository 
- Mission XML Documenation
- Claude AI
- Chat GPT

