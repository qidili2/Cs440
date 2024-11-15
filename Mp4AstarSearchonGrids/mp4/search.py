import heapq

def best_first_search(starting_state):
    # TODO(III): You should copy your code from MP3 here
    visited_states = {starting_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, starting_state)

    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------

    while frontier:
        current_state = heapq.heappop(frontier)
        if current_state.is_goal():
            return backtrack(visited_states, current_state)
        
        # Explore neighbors
        for neighbor in current_state.get_neighbors():
            neighbor_distance = neighbor.dist_from_start

            if neighbor not in visited_states or neighbor_distance < visited_states[neighbor][1]:
                visited_states[neighbor] = (current_state, neighbor_distance)
                heapq.heappush(frontier, neighbor)
    # ------------------------------
    
    # if you do not find the goal return an empty list
    return []

def backtrack(visited_states, goal_state):
    # TODO(III): You should copy your code from MP3 here
    path = []
    # Your code here ---------------
    current_state = goal_state
    while current_state is not None:
        path.append(current_state)
        current_state = visited_states[current_state][0]  
    
    path.reverse()
    # ------------------------------
    return path