import random
from tqdm import tqdm
import numpy as np
import statistics
import pandas as pd
import time
from pprint import PrettyPrinter
import pygame
import sys
from pygame_utils import (
    WHITE,
    START_STATE,
    GOAL_STATE,
    draw_maze,
    generate_maze,
)
from constants import *

screen = None
cell_size = None
start_state = None
goal_state = None
grid_size = None
obstacles = None


def generate_dynamics(
    grid_size: tuple, goal_state: tuple, obstacles: list[tuple]
) -> dict:
    p = dict()
    state = 0
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            p_inner = dict()
            # left action OK
            next_state = (col - 1, row)
            if next_state not in obstacles and col - 1 >= 0:
                p_inner[LEFT] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]
            # right action OK
            next_state = (col + 1, row)
            if next_state not in obstacles and col + 1 < grid_size[0]:
                p_inner[RIGHT] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]
            # up action OK
            next_state = (col, row - 1)
            if next_state not in obstacles and row - 1 >= 0:
                p_inner[UP] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]
            # down action OK
            next_state = (col, row + 1)
            if next_state not in obstacles and row + 1 < grid_size[1]:
                p_inner[DOWN] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]

            if (col, row) == goal_state:
                p_inner = dict()

            p[(col, row)] = p_inner
            state += 1

    return p


def validate_dynamics(obstacles, p, grid_size):
    errors = []
    for s in p.keys():
        for a in p[s].keys():
            if (
                p[s][a][1] in obstacles
                or p[s][a][1][0] >= grid_size[0]
                or p[s][a][1][1] >= grid_size[1]
            ):
                errors.append((s, a))

    if len(errors):
        for e in errors:
            print(f"In state {e[0]}, taking action {e[1]} leads to obstacle or wall")
        raise TypeError


def random_policy(state: tuple, p: dict) -> int:
    available_actions = list(p[state].keys())
    if len(available_actions):
        return random.choice(available_actions)
    else:
        raise TypeError(f"No available actions to choose from in state {state}")


def greedy_policy(state: tuple, p: dict, value_func: dict) -> int:
    available_actions = list(p[state].keys())
    if not len(available_actions):
        raise TypeError(f"No available actions to choose from in state {state}")

    max_value = 0
    max_act = random.choice(available_actions)
    for act in available_actions:
        next_state = p[state][act][1]
        if value_func[next_state] > max_value:
            max_value = value_func[next_state]
            max_act = act
        if next_state == GOAL_STATE:
            return act

    # print(f"{state} -> {p[state][max_act][1]} {max_act}")
    return max_act


def generate_episode(policy, dynamics, value_func):
    episode = []
    terminated = False
    state = START_STATE
    count = 0
    while not terminated:
        action = policy(state, dynamics, value_func)
        next_state = dynamics[state][action][1]
        reward = 1 if next_state == GOAL_STATE else 0
        episode.append((state, action, reward))
        terminated = dynamics[state][action][2]
        state = next_state
        count += 1
        if count > 100:
            terminated = True

    return episode


def eps_greedy_policy(state: tuple, p: dict, value_func: dict, eps: float):
    available_actions = list(p[state].keys())
    if not len(available_actions):
        raise TypeError(f"No available actions to choose from in state {state}")

    # exploitation action
    exploit_act = available_actions[0]
    max_value = 0
    for act in available_actions:
        next_state = p[state][act][1]
        if value_func[next_state] > max_value:
            max_value = value_func[next_state]
            exploit_act = act
        if next_state == GOAL_STATE:
            exploit_act = act

    # exploration action
    explore_act = random.choice(available_actions)

    if random.random() < eps:
        return explore_act
    else:
        return exploit_act


def optimize_policy_every_visit_mc(dynamics):
    gamma = 0.2
    policy = greedy_policy
    value_func = {s: 0 for s in dynamics.keys()}
    returns = {s: [] for s in dynamics.keys()}
    policy_changed = True
    actions_taken = []
    previous_actions_taken = []
    num_iterations = 0
    while policy_changed:
        episode = generate_episode(policy, dynamics, value_func)
        goodness = 0
        for s, a, r in reversed(episode):
            goodness = gamma * goodness + r
            returns[s].append(goodness)
            value_func[s] = sum(returns[s]) / len(returns[s])
            actions_taken.append(a)

        hard_policy = lambda s: policy(s, dynamics, value_func)
        screen.fill(WHITE)
        draw_maze(
            OBSTACLES_DEFAULT,
            hard_policy,
            dynamics,
            show_policy=True,
            value_func=value_func,
        )
        pygame.display.update()
        pygame.image.save(screen, f"{num_iterations}.jpg")
        if actions_taken == previous_actions_taken:
            policy_changed = False
            with open("optimal-actions.csv", "a") as f:
                f.write(
                    ",".join(map(str, reversed(actions_taken)))
                    + f",{num_iterations+1}\n"
                )

        previous_actions_taken = actions_taken
        actions_taken = []
        num_iterations += 1

    return value_func


def main():
    global screen, cell_size, start_state, goal_state, grid_size, obstacles
    grid_size = (30, 30)
    obstacles = generate_maze(grid_size[0] // 2)
    print(obstacles)
    start_state = (1, 1)
    goal_state = (grid_size[0] - 1, grid_size[1] - 1)

    p = generate_dynamics(grid_size, goal_state, obstacles)
    validate_dynamics(obstacles, p, grid_size)
    pp = PrettyPrinter(indent=2)
    pp.pprint(p)
    return

    # for _ in tqdm(range(5000)):
    #     value_func = optimize_policy_every_visit_mc(p)

    # df = pd.read_csv("optimal-actions.csv", on_bad_lines="warn", dtype=np.int32)
    # print(f"mean num iterations = {statistics.mean(df[df.columns[-1]])}")
    # print(f"std dev = {statistics.stdev(df[df.columns[-1]])}")
    # optimal_polices = df[df.columns[:-1]]
    # print(optimal_polices.drop_duplicates())

    cell_size = int(-0.03 * (5 - grid_size[0]) ** 2 + 50)
    print(cell_size)
    screen_size = (cell_size * grid_size[0], cell_size * grid_size[1])
    print(screen_size)

    # CELL_SIZE = 50
    # SCREEN_SIZE = (250*2+1, 250*2 +1)
    screen = pygame.display.set_mode(screen_size)
    running = True
    redraw = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if redraw:
            screen.fill(WHITE)
            draw_maze(
                screen,
                cell_size,
                (1, 1),
                (grid_size[0] - 1, grid_size[0] - 1),
                grid_size,
                obstacles,
                random_policy,
                None,
                show_policy=False,
                value_func=None,
            )
            pygame.display.update()
            redraw = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
