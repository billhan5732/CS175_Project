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

We will evaluate our model based on a few key metrics:
- Speed/Lap Time
- Staying on Track
- Track Progress (not idling for too long)