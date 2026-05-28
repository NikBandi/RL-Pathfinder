import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10

class Game:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = None

        self.head = Point(self.w / 2, self.h / 2)
        self.player = [self.head]

        self.score = 0
        self.goal = None
        self._place_goal()

        self.frame_iteration = 0


    def _place_goal(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.goal = Point(x, y)
        if self.goal in self.player:
            self._place_goal()


    def play_step(self, action):

        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.player.insert(0, self.head)
        self.player = [self.head]

        # 3. check if game over
        game_over = False
        reward = 0

        if self.is_collision() or self.frame_iteration> (100 * max(1, self.score)):
            game_over = True
            reward -= 10
            return reward, game_over, self.score

        # 4. place new goal or just move
        if self.head == self.goal:
            self.score += 1
            reward += 10
            self._place_goal()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt = None):
        # hits boundary
        if pt is None:
            pt = self.head

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.goal.x, self.goal.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):

        import numpy as np
        action_idx = np.argmax(action)

        # action moves = [up, down, left, right]

        if action_idx == 0:
            self.direction = Direction.UP
        elif action_idx == 1:
            self.direction = Direction.DOWN
        elif action_idx == 2:
            self.direction = Direction.LEFT
        elif action_idx == 3:
            self.direction = Direction.RIGHT


        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
