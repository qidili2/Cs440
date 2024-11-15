# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# # TODO: VI
# def astar(maze):
#     """使用 A* 算法进行搜索。"""
#     start_state = maze.get_start()
#     frontier = []  # 使用优先队列（堆）
#     visited = set()  # 存储已访问状态（避免重复访问）
#     heapq.heappush(frontier, (start_state.dist_from_start, start_state))  # (cost, state)

#     explored_count = 0

#     visited_states = {}  # 用于回溯路径

#     while frontier:
#         _, current_state = heapq.heappop(frontier)
        
#         # 如果已经访问过，跳过
#         if current_state.state in visited:
#             continue

#         visited.add(current_state.state)  # 标记为已访问
#         explored_count += 1
#         # print(f"Exploring state: {current_state.state}")

#         if current_state.is_goal():
#             path = backtrack(visited_states, current_state)
#             # print(f"Path found! Length: {len(path)}, Total Cost: {current_state.dist_from_start}")
#             return path

#         # 遍历当前状态的所有邻居
#         for neighbor in current_state.get_neighbors():
#             # print(f"Checking neighbor: {neighbor.state}")

#             if neighbor.state not in visited:
#                 visited_states[neighbor.state] = current_state  # 存储父节点信息
#                 heapq.heappush(frontier, (neighbor.dist_from_start+ neighbor.compute_heuristic(), neighbor))  # 加入 frontier

#     return None



# # Go backwards through the pointers in visited_states until you reach the starting state
# # NOTE: the parent of the starting state is None
# # TODO: VI
# def backtrack(visited_states, current_state):
#     """回溯路径，从终点到起点。"""
#     path = []

#     # 使用父节点关系回溯路径
#     while current_state is not None:
#         path.append(current_state)
#         # 直接获取父状态
#         current_state = visited_states.get(current_state.state)

#     path.reverse()  # 逆序路径，从起点到终点
#     print("Path found:")
#     for state in path:
#         print(f"State: {state.state}, Cost: {state.dist_from_start}")
#     print(f"Total Path Length: {len(path)}")
#     return path
# TODO: VI
def astar(maze):
    starting_state = maze.get_start()
    visited_states = {starting_state: (None, 0)}

    frontier = []
    heapq.heappush(frontier, starting_state)

    while frontier:
        
        # pop best
        current_state = heapq.heappop(frontier)

        # if found turn back
        if current_state.is_goal():
            return backtrack(visited_states, current_state)

        # all neighbors
        for neighbor in current_state.get_neighbors():
            neighbor_dist = neighbor.dist_from_start

            if neighbor not in visited_states or neighbor_dist < visited_states[neighbor][1]:
                visited_states[neighbor] = (current_state, neighbor_dist)
                heapq.heappush(frontier, neighbor)

    return None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    while current_state:
        path.append(current_state)
        current_state = visited_states[current_state][0]
    return path[::-1] 