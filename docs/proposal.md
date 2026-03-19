---
layout: default
title:  Proposal
---

## Summary of the Project
Boat racing is a common racing minigame in Minecraft. Each player a boat and must navigate the winding track and environment on ice via W A S D keys. Players will race for the fastest track time, and the fastest times are listed ascending on the leaderboard. Our project will train an AI agent to play the game and explore the efficiencies of different RL algorithms by comparing the times of each. These comparisons will help us distinguish what skill level the AI agent is most similar to, and thus its level of success in boat racing.

This project aims to use AI/ML to control a boat in minecraft boat racing.
This requires a lot of human precision and acute motor skills which makes it great for an RL AI agent.

## Project goals

### Minimum Goal

Our minimum goal is to control an AI minecraft agent to drive a boat on a race track without going off the ice while making reasonable progress.
It should be able to learn to beat at least one of the test tracks we train it on.
### Realistic Goal

Realistically, we aim to implement an AI agent that learns to complete a lap on an ice race track.
Our initial idea is to have 1-3 tracks (ideally procedurally generated) the agent is trained on and 1-2 novel tracks to test it on.
Ideally our agent is able to beat all 5.
### Moonshot Goal

Make it speed run the particular ice track to find the optimal racing lines.
Have it consider other racers that could potentially bump and push it around.

## AI/ML Algorithms

We will mainly be using Reinforcement Learning strategies with this agent. Some strategies/algorithms we hypothesize that we will use are as follows:
- Proximal Policy Optimization
- Deep Q-Learning (based on prior "Stealth Escape" project)




## Evaluation Plan

We will evaluate our agent using the following metrics:

- **Speed (Lap Time)**
  - Measure total time to complete a track
  - Track average and best lap times across multiple runs
  - Compare early vs later training performance

- **Track Completion**
  - Measure percentage of runs where the agent successfully finishes the track
  - Use a fixed time limit to determine success or failure
  - Reflects overall reliability of the agent

- **Stability (Staying on Track)**
  - Track number of falls or resets per run
  - Measure time spent off-track
  - Evaluate how consistently the agent maintains control

- **Progress Efficiency**
  - Measure checkpoints reached per episode
  - Track forward progress per timestep
  - Ensure the agent is not idling or exploiting unintended behaviors

- **Learning Progress**
  - Monitor performance trends over training 
  - Evaluate whether the agent improves over time


## Meetings
Jan 23rd, 2026 3:45pm - First meeting on zoom.
Feb 13rd, 2026 11:15am - Second meeting on zoom.
Mar 6th, 2026 10:15am - Third meeting on zoom.

## AI Tool Usage
No AI Tool has been utilzied so far.