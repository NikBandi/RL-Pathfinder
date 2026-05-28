# Pathfinder — Reinforcement Learning Agent

A deep Q-learning agent that learns to navigate a 2D grid and reach a goal as many times as possible before hitting a wall.

## How it works

The agent uses a neural network to learn which direction to move based on its current state. It gets rewarded for reaching the goal and penalized for hitting a wall. Over time it learns to navigate efficiently.

The state the agent sees each frame:
- Danger in each of the 4 directions (wall ahead)
- Current movement direction
- Whether the goal is above, below, left, or right

The action space is 4 moves: up, down, left, right.

## Training Results

Before:

<img width="638" height="510" alt="Screen Recording 2026-05-28 132551" src="https://github.com/user-attachments/assets/67bed9f2-3658-4228-a96f-cfb52e81d8f9" />

After:

<img width="636" height="514" alt="trained" src="https://github.com/user-attachments/assets/dd14c5aa-dc9e-432c-b9b2-0d45e0e54542" />


## Project structure

```
├── agent_train.py    # Training loop and agent logic
├── game.py           # pygame game environment
├── model.py          # Neural network and Q-trainer
├── play.py           # Run the trained agent without training
├── plot_helper.py    # Plot utility
├── arial.ttf         # Font for the game window
└── model/
    └── model.pth     # Saved model weights (created after first record)
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Train the agent**

```bash
python agent_train.py
```

A pygame window will open showing the agent playing in real time. Training progress is printed to the console. The model saves automatically whenever a new high score is reached.

**3. Watch the trained agent**

```bash
python play.py
```

Loads the saved model and runs it without any training.

## Model

The neural network is a 4-layer fully connected network:

```
Input (12) → Hidden (256) → Hidden (256) → Hidden (256) → Output (4)
```

The model is saved automatically to `model/model.pth` whenever a new high score is reached.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Gamma (discount) | 0.725 |
| Epsilon decay | 80 - n_games |
| Batch size | 1000 |
| Max memory | 100,000 |

## Requirements

- Python 3.11
- torch
- pygame
- numpy
- matplotlib
