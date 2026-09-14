"""
snake_game.py
-------------
Core Snake game logic built with Pygame.
Separates rendering from game logic so the Gymnasium wrapper
can run headless (no window) during training.
"""

import pygame
import numpy as np
from enum import Enum
from collections import namedtuple

# -- Constants ----------------------------------------------------------------
GRID_SIZE   = 20          # 20x20 grid as specified in the proposal
CELL_SIZE   = 30          # pixels per cell (only used when rendering)
WINDOW_SIZE = GRID_SIZE * CELL_SIZE   # 600x600 px window

FPS = 15                  # frames/sec for human-playable speed

# Colors (RGB)
BLACK  = (0,   0,   0)
WHITE  = (255, 255, 255)
GREEN  = (0,   200, 0)
DGREEN = (0,   140, 0)
RED    = (200, 0,   0)
GRAY   = (40,  40,  40)

# -- Direction enum ------------------------------------------------------------
class Direction(Enum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3

Point = namedtuple("Point", ["x", "y"])   # grid coordinates

# -- SnakeGame class -----------------------------------------------------------
class SnakeGame:
    """
    Pure game logic + optional Pygame rendering.

    Parameters
    ----------
    render : bool
        If True, opens a Pygame window. Set False during RL training.
    seed : int | None
        Random seed for reproducible food placement.
    """

    def __init__(self, render: bool = False, seed: int | None = None):
        self.render_mode = render
        self.rng = np.random.default_rng(seed)

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake – RL Agent")
            self.clock  = pygame.time.Clock()
            self.font   = pygame.font.SysFont("monospace", 22)

        self.reset()

    # -- Reset -----------------------------------------------------------------
    def reset(self, seed=None):
        """Start a new episode. Returns nothing – call get_state() afterward."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
        self.direction = Direction.RIGHT
        # Snake starts as 3 cells
        self.snake = [
            Point(cx,     cy),
            Point(cx - 1, cy),
            Point(cx - 2, cy),
        ]
        self.score      = 0
        self.frame_iter = 0          # cap episodes to avoid infinite loops
        self._place_food()

    # -- Food placement ---------------------------------------------------------
    def _place_food(self):
        while True:
            x = int(self.rng.integers(0, GRID_SIZE))
            y = int(self.rng.integers(0, GRID_SIZE))
            pt = Point(x, y)
            if pt not in self.snake:
                self.food = pt
                break

    # -- Step -----------------------------------------------------------------
    def step(self, action: int):
        """
        Advance the game by one frame.

        Actions (relative to current heading):
            0 = go straight
            1 = turn right
            2 = turn left

        Returns
        -------
        reward : float
        game_over : bool
        score : int
        """
        self.frame_iter += 1

        # Handle quit events when rendering
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # -- Compute new direction from relative action -------------------
        #   Direction order: RIGHT(0) → DOWN(1) → LEFT(2) → UP(3)
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if action == 0:                    # straight
            new_dir = clockwise[idx]
        elif action == 1:                  # right turn (clockwise)
            new_dir = clockwise[(idx + 1) % 4]
        else:                              # left turn (counter-clockwise)
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        # -- Move head ----------------------------------------------------
        hx, hy = self.snake[0]
        if   self.direction == Direction.RIGHT: hx += 1
        elif self.direction == Direction.LEFT:  hx -= 1
        elif self.direction == Direction.DOWN:  hy += 1
        elif self.direction == Direction.UP:    hy -= 1
        new_head = Point(hx, hy)

        self.snake.insert(0, new_head)

        # -- Collision check ----------------------------------------------
        game_over = False
        reward    = 0.0

        if self._is_collision(new_head):
            game_over = True
            reward    = -10.0
            self.snake.pop()
            return reward, game_over, self.score

        # Kill episodes that run too long (avoids the agent spinning forever)
        if self.frame_iter > 100 * len(self.snake):
            game_over = True
            reward    = -10.0
            self.snake.pop()
            return reward, game_over, self.score

        # -- Eat food ------------------------------------------------------
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()           # tail stays → snake grows
        else:
            self.snake.pop()             # normal move: remove tail

        if self.render_mode:
            self._draw()

        return reward, game_over, self.score

    # -- Collision helper --------------------------------------------------
    def _is_collision(self, pt: Point | None = None) -> bool:
        if pt is None:
            pt = self.snake[0]
        # Wall
        if pt.x < 0 or pt.x >= GRID_SIZE or pt.y < 0 or pt.y >= GRID_SIZE:
            return True
        # Self
        if pt in self.snake[1:]:
            return True
        return False

    # -- 11-dimensional state vector -------------------------------------------
    def get_state(self) -> np.ndarray:
        """
        Returns a length-11 binary numpy array describing the current state.

        Layout (matches the proposal exactly):
            [0]  danger straight
            [1]  danger right
            [2]  danger left
            [3]  moving right
            [4]  moving down
            [5]  moving left
            [6]  moving up
            [7]  food to the left of head
            [8]  food to the right of head
            [9]  food above head
            [10] food below head
        """
        head = self.snake[0]
        d    = self.direction

        # One step ahead in each relative direction
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx       = clockwise.index(d)

        def ahead(n):
            """Point n steps in relative direction (0=straight,1=right,2=left)"""
            dirs  = clockwise
            turns = [0, 1, -1]          # straight, right, left
            nd    = dirs[(idx + turns[n]) % 4]
            hx, hy = head
            if   nd == Direction.RIGHT: hx += 1
            elif nd == Direction.LEFT:  hx -= 1
            elif nd == Direction.DOWN:  hy += 1
            elif nd == Direction.UP:    hy -= 1
            return Point(hx, hy)

        danger_straight = int(self._is_collision(ahead(0)))
        danger_right    = int(self._is_collision(ahead(1)))
        danger_left     = int(self._is_collision(ahead(2)))

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            int(d == Direction.RIGHT),
            int(d == Direction.DOWN),
            int(d == Direction.LEFT),
            int(d == Direction.UP),
            int(self.food.x < head.x),   # food left
            int(self.food.x > head.x),   # food right
            int(self.food.y < head.y),   # food above  (y increases downward)
            int(self.food.y > head.y),   # food below
        ], dtype=np.float32)

        return state

    # -- Rendering -------------------------------------------------------------
    def _draw(self):
        self.screen.fill(BLACK)

        # Grid lines (subtle)
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_SIZE, y))

        # Snake
        for i, pt in enumerate(self.snake):
            color = GREEN if i > 0 else DGREEN
            rect  = pygame.Rect(pt.x * CELL_SIZE + 1, pt.y * CELL_SIZE + 1,
                                 CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Food
        fx = self.food.x * CELL_SIZE + CELL_SIZE // 2
        fy = self.food.y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, RED, (fx, fy), CELL_SIZE // 2 - 2)

        # Score HUD
        score_surf = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_surf, (8, 8))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()


# -- Quick human-playable test -------------------------------------------------
# NOTE: This block is ONLY executed when running snake_game.py directly.
# It is never called during RL training. All changes here are isolated
# to human play and have zero effect on the agent, environment, or rewards.
if __name__ == "__main__":
    game = SnakeGame(render=True)
    print("Use arrow keys to play. Close the window to quit.")
 
    # Desired absolute direction - held between frames so the snake
    # keeps moving in the last pressed direction (standard Snake behaviour).
    # Starts as RIGHT to match the snake's initial heading.
    desired_dir = Direction.RIGHT
 
    while True:
        # -- Event handling ------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                # Only update desired_dir if not a 180° reversal into the body
                if event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    desired_dir = Direction.RIGHT
                elif event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    desired_dir = Direction.LEFT
                elif event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    desired_dir = Direction.UP
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    desired_dir = Direction.DOWN
 
        # -- Convert absolute desired direction → relative action ----------
        # The game engine works in relative actions (0=straight,1=right,2=left)
        # so we translate here. This keeps step() unchanged for the RL agent.
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        cur_idx  = clockwise.index(game.direction)
        want_idx = clockwise.index(desired_dir)
        diff     = (want_idx - cur_idx) % 4
 
        if diff == 0:
            action = 0   # straight
        elif diff == 1:
            action = 1   # right turn (clockwise)
        elif diff == 3:
            action = 2   # left turn (counter-clockwise)
        else:
            # diff == 2 means 180° reversal — ignore and go straight instead
            action = 0
 
        # -- Step and handle episode end -----------------------------------
        reward, done, score = game.step(action)
        if done:
            print(f"Game over – score: {score}")
            game.reset()
            desired_dir = Direction.RIGHT   # reset to match new initial heading
 

