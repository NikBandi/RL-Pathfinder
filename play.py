import torch
from game import Game, Direction, Point
from model import Linear_QNet
import numpy as np

def get_state(game):
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
        game.is_collision(point_u),
        game.is_collision(point_d),
        game.is_collision(point_r),
        game.is_collision(point_l),
        dir_U, dir_D, dir_R, dir_L,
        game.goal.y < game.head.y,
        game.goal.y > game.head.y,
        game.goal.x < game.head.x,
        game.goal.x > game.head.x,
    ]

    return np.array(state, dtype=int)


def play():
    model = Linear_QNet(12, 256, 256, 256, 4)
    model.load()
    model.eval()

    game = Game()
    record = 0

    while True:
        state = get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            prediction = model(state_tensor)
            move = torch.argmax(prediction).item()

        action = [0, 0, 0, 0]
        action[move] = 1

        reward, done, score = game.play_step(action)

        if done:
            game.reset()
            if score > record:
                record = score
            print('Score', score, 'Record', record)


if __name__ == '__main__':
    play()