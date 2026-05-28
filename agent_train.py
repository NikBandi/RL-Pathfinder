import torch
import random
import numpy as np

from collections import deque
from game import Game, Direction, Point

from model import Linear_QNet, QTrainer

from plot_helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

learning_rate = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0    # randomness
        self.gamma = 0.725      # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)      # popleft()

        # TODO: model, trainer
        self.model = Linear_QNet(12, 256, 256, 256, 4) #TODO
        self.trainer = QTrainer(self.model, learning_rate=learning_rate, gamma= self.gamma)

        pass

    def get_state(self, game):

        head = game.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_L = game.direction == Direction.LEFT
        dir_R = game.direction == Direction.RIGHT
        dir_U = game.direction == Direction.UP
        dir_D = game.direction == Direction.DOWN

        state = [
            # Danger Up
            (game.is_collision(point_u)),

            # Danger Down
            (game.is_collision(point_d)),

            # Danger Right
            (game.is_collision(point_r)),

            #Danger Left
            (game.is_collision(point_l)),

            # move direction
            dir_U,
            dir_D,
            dir_R,
            dir_L,

            # Goal location
            game.goal.y < game.head.y,  # goal is above the player
            game.goal.y > game.head.y,  # goal is below the player
            game.goal.x < game.head.x, # goal is to the left of the player
            game.goal.x > game.head.x, # goal is to the right of the player
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is exceeded
        pass

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Creates a list of Tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

        pass


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        pass


    def get_action(self, state):
        # random moves
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    AI_game = Game()

    while True:
        # get current state
        state_old = agent.get_state(AI_game)

        # get move
        next_move = agent.get_action(state_old)

        # execute move
        reward, done, score = AI_game.play_step(next_move)
        state_new = agent.get_state(AI_game)

        # train short memory
        agent.train_short_memory(state_old, next_move, reward, state_new, done)
        agent.remember(state_old, next_move, reward, state_new, done)

        # train longer memory, and plot results
        if done:
            AI_game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Record", record)

            plot_scores.append(score)
            total_score += score
            plot_avg_scores.append((total_score / agent.n_games))

            plot(plot_scores, plot_avg_scores)

            # TODO: plot

    pass

if __name__ == '__main__':
    train()