import matplotlib.pyplot as plt
import math
from collections.abc import Callable
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
max_reward = 100
obstacle_reward = -1000

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
            if col - 1 >= 0:
                p_inner[LEFT] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]
            # right action OK
            next_state = (col + 1, row)
            if col + 1 < grid_size[0]:
                p_inner[RIGHT] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]
            # up action OK
            next_state = (col, row - 1)
            if row - 1 >= 0:
                p_inner[UP] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]
            # down action OK
            next_state = (col, row + 1)
            if row + 1 < grid_size[1]:
                p_inner[DOWN] = [
                    1.0,
                    next_state,
                    True if next_state == goal_state else False,
                ]

            if (col, row) == goal_state:
                p_inner = dict()

            reward = 0
            if (col, row) in obstacles:
                reward = obstacle_reward

            if (col, row) == goal_state:
                reward = max_reward

            p_inner["reward"] = reward

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

    return max_act


def generate_episode(policy, dynamics, value_func, eps, truncation_limit=100):
    episode = []
    terminated = False
    truncated = False
    state = start_state
    prev_state = start_state
    count = 0
    visited_states = []
    while not terminated:
        visited_states.append(state)
        action = policy(state, dynamics, value_func, eps, prev_state, visited_states)
        next_state = dynamics[state][action][1]
        reward = dynamics[state]["reward"]
        episode.append((state, action, reward))
        terminated = dynamics[state][action][2]
        prev_state = state
        state = next_state
        count += 1
        if count > truncation_limit:
            terminated = True
            truncated = True

    return episode, truncated


def eps_greedy_policy(state: tuple, p: dict, value_func: dict, eps: float):
    available_actions = list(p[state].keys())
    if not len(available_actions):
        raise TypeError(f"No available actions to choose from in state {state}")

    # exploitation action
    exploit_act = random.choice(available_actions)
    max_value = 0
    for act in available_actions:
        next_state = p[state][act][1]
        if value_func[next_state] > max_value:
            max_value = value_func[next_state]
            exploit_act = act

    # exploration action
    explore_act = random.choice(available_actions)

    if random.random() < eps:
        return explore_act
    else:
        return exploit_act

def eps_greedy_explore_policy(state: tuple, p: dict, value_func: dict, eps: float, prev_state=None, visited_states=[]):
    available_actions = list(p[state].keys())[:-1]
    if not len(available_actions):
        raise TypeError(f"No available actions to choose from in state {state}")

    if len(available_actions) == 1:
        return available_actions[0]

    # if len(available_actions) == 2 and prev_state:
    #     if p[state][available_actions[0]][1] == prev_state:
    #         return available_actions[1]
    #     elif p[state][available_actions[1]][1] == prev_state:
    #         return available_actions[0]

    rand = random.random()

    # exploitation action
    exploit_act = random.choice(available_actions)
    max_value = -9999999999
    # actions_not_visited = []
    for act in available_actions:
        next_state = p[state][act][1]
        if next_state in visited_states:
            continue
        if value_func[next_state] > max_value:
            max_value = value_func[next_state]
            exploit_act = act
        # if we haven't visited this state before
        # if value_func[next_state] == 0:
        #     actions_not_visited.append(act)

    # if eps == 1.0 and len(actions_not_visited) and rand < 0.5:
    #     return random.choice(actions_not_visited)

    # exploration action
    if rand < eps:
        return random.choice(available_actions)
    # exploitation action
    else:
        return exploit_act


def optimize_policy_every_visit_mc(dynamics, gamma, debug=False):
    eps = 1.0
    eps_decay_rate = 0.001
    previous_goodness = 0
    policy = eps_greedy_explore_policy
    value_func = {s: 0 for s in dynamics.keys()}
    best_value_fn = {s: 0 for s in dynamics.keys()}
    state_count = {s: 0 for s in dynamics.keys()}
    value_func[goal_state] = 100
    policy_changed = True
    num_iterations = 0
    truncated = False
    truncation_step = 20000
    truncation_limit = truncation_step
    exploration_complete = False
    episode = tuple()
    max_goodness = 0
    goodness = 0

    epsilon_plot = []
    goodness_plot = []
    diff_plot = []

    while policy_changed:
        episode, truncated = generate_episode(policy, dynamics, value_func, eps, truncation_limit)
        if truncated:
            # print("episode truncated")
            # truncation_limit += truncation_step
            continue

        old_values = [v for v in value_func.values()]
        old_value_fn = value_func
        goodness = 0
        for s, _, r in reversed(episode):
            goodness = (gamma * goodness) + r
            state_count[s] += 1
            value_func[s] = (value_func[s] + goodness) / state_count[s]

        # if goodness == 2.5e-323 and truncation_limit > truncation_step * 2:
        #     truncation_limit -= truncation_step
        if goodness > max_goodness:
            max_goodness = goodness
            best_value_fn = old_value_fn

        new_values = [v for v in value_func.values()]
        diff = math.sqrt(np.abs(np.mean(np.subtract(old_values, new_values))))
        print(f"episode {num_iterations} complete in {len(episode)} steps, G={goodness}, e={eps}")

        if debug:
            screen.fill(WHITE)
            draw_maze(
                screen,
                cell_size,
                start_state,
                goal_state,
                grid_size,
                obstacles,
                policy=lambda s: policy(s, dynamics, value_func, 0.0),
                dynamics=dynamics,
                show_policy=True,
                value_func=value_func,
                path=tuple(e[0] for e in episode)
            )
            pygame.display.update()
            for v in value_func.values():
                if v == 2.5e-323:
                    raise OverflowError("Increase max reward or gamma")

        if exploration_complete == False:
            for state in dynamics.keys():
                if value_func[state] == 0:
                    break
            else:
                exploration_complete = True

        if exploration_complete:
            eps *= (1 - eps_decay_rate)

        num_iterations += 1
        epsilon_plot.append(eps)
        goodness_plot.append(goodness)
        diff_plot.append(diff)

        if len(diff_plot) > 50 and np.mean(diff_plot[-50:]) < 0.009:
            policy_changed = False
        # if  goodness > 2.5e-5 and goodness == previous_goodness:
        #     policy_changed = False
            #with open("optimal-actions.csv", "a") as f:
            #    f.write(
            #        ",".join(map(str, reversed(actions_taken)))
            #        + f",{num_iterations+1}\n"
            #    )
        previous_goodness = goodness

    screen.fill(WHITE)
    path = {e[0]: e[1] for e in episode}
    print(path)
    draw_maze(
        screen,
        cell_size,
        start_state,
        goal_state,
        grid_size,
        obstacles,
        policy=lambda s: policy(s, dynamics, value_func, 0.0),
        dynamics=dynamics,
        show_policy=True,
        value_func=value_func,
        path=path
    )
    pygame.display.update()

    ax1 = plt.subplot(311)
    plt.plot(goodness_plot)
    plt.tick_params('x', labelsize=6)
    ax1.set_ylabel("Goodness")
    plt.title("Training Metrics")

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(diff_plot)
    plt.tick_params('x', labelbottom=False)
    ax2.set_ylabel("Root Mean V(s) diff")

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(epsilon_plot)
    ax3.set_ylabel("Epsilon")
    plt.xlabel("Episode Number")

    plt.show()

    print(f"best goodness {max_goodness}")

    pygame.image.save(screen, f"output/{num_iterations}.jpg")
    print(f"convergence in {num_iterations} iterations")
    return value_func


def main():
    global screen, cell_size, start_state, goal_state, grid_size, obstacles
    sys.setrecursionlimit(10000)
    grid_size = (30, 30)
    obstacles = generate_maze(grid_size[0] // 2)
    start_state = (0,0)
    goal_state = (grid_size[0]-1, grid_size[1]-1)
    cell_size = int(-(grid_size[0]/1000) * (5 - grid_size[0]) ** 2 + 50)
    if cell_size < 0:
        cell_size = 10
    screen_size = (cell_size * grid_size[0], cell_size * grid_size[1])
    screen = pygame.display.set_mode(screen_size)

    # original problem
    # obstacles = OBSTACLES_DEFAULT
    # start_state = (0, 0)
    # goal_state = (4, 4)

    p = generate_dynamics(grid_size, goal_state, obstacles)
    # validate_dynamics(obstacles, p, grid_size)
    pp = PrettyPrinter(indent=2)
    pp.pprint(p)

    # for _ in tqdm(range(5000)):
    #     value_func = optimize_policy_every_visit_mc(p)

    # df = pd.read_csv("optimal-actions.csv", on_bad_lines="warn", dtype=np.int32)
    # print(f"mean num iterations = {statistics.mean(df[df.columns[-1]])}")
    # print(f"std dev = {statistics.stdev(df[df.columns[-1]])}")
    # optimal_polices = df[df.columns[:-1]]
    # print(optimal_polices.drop_duplicates())

    screen.fill(WHITE)
    draw_maze(
        screen,
        cell_size,
        start_state,
        goal_state,
        grid_size,
        obstacles,
        policy=random_policy,
        dynamics=None,
        show_policy=False,
        value_func=None,
    )
    pygame.display.update()

    value_func = optimize_policy_every_visit_mc(p, 0.9, debug=False)
    print(value_func)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
