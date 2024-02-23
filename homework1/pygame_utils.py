import random
import pygame
from constants import *

pygame.init()

def generate_maze(grid_size):
    # Initialize the grid with walls ('W') and paths ('P')
    maze = [['W' for _ in range(grid_size * 2 + 1)] for _ in range(grid_size * 2 + 1)]
    
    for row in range(grid_size * 2 + 1):
        for col in range(grid_size * 2 + 1):
            if row % 2 == 1 and col % 2 == 1:
                maze[row][col] = 'P'
    
    # Start position
    start_x, start_y = 1, 1
    # End position
    end_x, end_y = grid_size * 2 - 1, grid_size * 2 - 1

    # Mark start and end
    maze[start_x][start_y] = 'S'
    maze[end_x][end_y] = 'E'
    maze[end_x-1][end_y] = ' '
    maze[end_x][end_y-1] = ' '
    
    def carve_maze(x, y):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < grid_size * 2 and 1 <= ny < grid_size * 2 and maze[nx][ny] == 'P':
                maze[nx][ny] = ' '
                maze[x + dx // 2][y + dy // 2] = ' '
                carve_maze(nx, ny)
    
    carve_maze(start_x, start_y)
    
    # Print the maze
    for row in maze:
        print(' '.join(row))

    obstacles = []
    for row in range(grid_size * 2 + 1):
        for col in range(grid_size * 2 + 1):
            if maze[row][col] == 'W':
                obstacles.append((col,row))

    return obstacles

def draw_maze(screen, cell_size, start_state, goal_state, grid_size, obstacles, policy=None, dynamics=None, show_policy=False, value_func=None):
    pygame.display.set_caption(f"{grid_size[0]}x{grid_size[1]} Maze")
    font = pygame.font.SysFont(None, 24)
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, WHITE, rect, 1)
            if (col, row) in obstacles:
                pygame.draw.rect(screen, OBSTACLE_COLOR, rect)
            elif show_policy:
                try:
                    direction = policy((col, row))
                    draw_arrow(screen, rect, direction)
                except TypeError:
                    pass
            if value_func and (col, row) not in obstacles:
                val = font.render(str(round(value_func[(col, row)], 3)), True, BLACK)
                screen.blit(val, rect)

    start_rect = pygame.Rect(start_state[0] * cell_size, start_state[1] * cell_size, cell_size, cell_size)
    goal_rect = pygame.Rect(goal_state[0] * cell_size, goal_state[1] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, BLACK, start_rect)
    pygame.draw.rect(screen, BLACK, goal_rect)
    text_start = font.render('Start', True, WHITE)
    text_goal = font.render('Goal', True, WHITE)
    screen.blit(text_start, start_rect)
    screen.blit(text_goal, goal_rect)


def draw_left_arrow(surface, rect, color):
    center_x, center_y = rect.center
    arrow_size = min(rect.width, rect.height) // 2
    left_point = (center_x - arrow_size // 2, center_y)
    top_point = (center_x + arrow_size // 2, center_y - arrow_size // 2)
    bottom_point = (center_x + arrow_size // 2, center_y + arrow_size // 2)

    point_list = [left_point, top_point, (center_x + arrow_size // 2, center_y), bottom_point]

    pygame.draw.polygon(surface, color, point_list)

def draw_right_arrow(surface, rect, color):
    center_x, center_y = rect.center
    arrow_size = min(rect.width, rect.height) // 2
    right_point = (center_x + arrow_size // 2, center_y)
    top_point = (center_x - arrow_size // 2, center_y - arrow_size // 2)
    bottom_point = (center_x - arrow_size // 2, center_y + arrow_size // 2)

    point_list = [right_point, top_point, (center_x - arrow_size // 2, center_y), bottom_point]

    pygame.draw.polygon(surface, color, point_list)

def draw_down_arrow(surface, rect, color):
    center_x, center_y = rect.center
    arrow_size = min(rect.width, rect.height) // 2
    bottom_point = (center_x, center_y + arrow_size // 2)
    left_point = (center_x - arrow_size // 2, center_y - arrow_size // 2)
    right_point = (center_x + arrow_size // 2, center_y - arrow_size // 2)

    point_list = [bottom_point, left_point, (center_x, center_y - arrow_size // 2), right_point]

    pygame.draw.polygon(surface, color, point_list)

def draw_up_arrow(surface, rect, color):
    center_x, center_y = rect.center
    arrow_size = min(rect.width, rect.height) // 2
    top_point = (center_x, center_y - arrow_size // 2)
    left_point = (center_x - arrow_size // 2, center_y + arrow_size // 2)
    right_point = (center_x + arrow_size // 2, center_y + arrow_size // 2)

    point_list = [top_point, left_point, (center_x, center_y + arrow_size // 2), right_point]

    pygame.draw.polygon(surface, color, point_list)

def draw_arrow(surface, rect, direction):
    if direction == LEFT:
        draw_left_arrow(surface, rect, (32,32,32))
    elif direction == RIGHT:
        draw_right_arrow(surface, rect, (64,64,64))
    elif direction == DOWN:
        draw_down_arrow(surface, rect, (128,128,128))
    elif direction == UP:
        draw_up_arrow(surface, rect, (96,96,96))


