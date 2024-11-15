# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy
def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int, int, int, int]], waypoint: Tuple[int, int]) -> bool:

    start_head, start_tail = alien.get_head_and_tail()
    alien.set_alien_pos(waypoint)
    end_head, end_tail = alien.get_head_and_tail()
    if start_head == end_head and start_tail == end_tail:
        return False
    if alien.is_circle():
        radius = alien.get_width() 
        for wall in walls:
            if does_circle_path_intersect_wall(start_head, waypoint, radius, wall):
                return True
    else:
        parallelogram = [start_head, start_tail, end_tail, end_head]
        parallelogram_edges = [(start_head,start_tail),(start_tail,end_tail),(end_tail,end_head),(end_head,start_head)]
        radius = alien.get_width()
        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            

            for edge in parallelogram_edges:
                if do_segments_intersect(edge, wall_segment):
                    return True
                
            for edge in parallelogram_edges:
                if point_segment_distance(wall_segment[0], edge) <= radius or point_segment_distance(wall_segment[1], edge) <= radius:
                    return True
                
            if is_point_in_polygon(wall_segment[0], parallelogram) or is_point_in_polygon(wall_segment[1], parallelogram):
                return True
    return False

def does_circle_path_intersect_wall(start: Tuple[int, int], end: Tuple[int, int], radius: float, wall: Tuple[int, int, int, int]) -> bool:
    wall_start = (wall[0], wall[1])
    wall_end = (wall[2], wall[3])
    
    if segment_distance((start, end), (wall_start, wall_end)) <= radius:
        return True
    return False


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    if alien.is_circle(): 
        alien_position = alien.get_centroid()
        alien_radius = alien.get_width() 
        
        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            
            distance = point_segment_distance(alien_position, wall_segment)

            if distance <= alien_radius:
                return True
        return False
    else: 
        alien_head, alien_tail = alien.get_head_and_tail()  
        alien_width = alien.get_width()  
        
        for wall in walls:
            wall_segment = ((wall[0], wall[1]), (wall[2], wall[3]))
            
            distance = segment_distance((alien_head, alien_tail), wall_segment)
            
            if distance <= alien_width:
                return True
        return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    head, tail = alien.get_head_and_tail()
    alien_width = alien.get_width()

    for point in [head, tail]:
        if (point[0] - alien_width <= 0 or point[0] + alien_width >= window[0] or
            point[1] - alien_width <= 0 or point[1] + alien_width >= window[1]):
            return False

    if alien.get_shape() == 'Horizontal':
        if head[0] + alien_width >= window[0] or tail[0] - alien_width <= 0:
            return False
    elif alien.get_shape() == 'Vertical':
        if head[1] + alien_width >= window[1] or tail[1] - alien_width <= 0:
            return False

    return True

def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a polygon or on its edge.
    
    Args:
        point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
        polygon (list): shape of vertices of the polygon.
    
    Returns:
        bool: True if the point is inside or on the edge of the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        if p1[1] == p2[1] and p1[1] == y and min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]):
            return True
        if p1[0] == p2[0] and p1[0] == x and min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]):
            return True

    px, py = polygon[0]
    for i in range(1, n + 1):
        qx, qy = polygon[i % n]
        
        if y > min(py, qy):
            if y <= max(py, qy):
                if x <= max(px, qx):
                    if py != qy:
                        xinters = (y - py) * (qx - px) / (qy - py) + px
                    if px == qx or x <= xinters:
                        inside = not inside
        px, py = qx, qy

    return inside

    # Check the point relative to each edge of the polygon
    signs = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        cross_product = cross_product_sign(p1, p2, p)
        
        # Check if the point is exactly on the current edge
        if cross_product == 0 and is_point_on_segment(p1, p2, p):
            return True
        
        signs.append(np.sign(cross_product))

    # If all cross products have the same sign, the point is inside
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)




def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    px, py = p
    (x1, y1), (x2, y2) = s

    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if line_len_sq == 0:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))

    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    def orientation(p, q, r):
        """Helper function to find orientation of ordered triplet (p, q, r)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:  # Collinear
            return 0
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def on_segment(p, q, r):
        """Check if point q lies on line segment pr."""
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    p1, q1 = s1
    p2, q2 = s2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases: collinear points
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0.0
    # 计算两段不相交时的最短距离
    return min(
        point_segment_distance(s1[0], s2),
        point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1),
        point_segment_distance(s2[1], s1),
    )


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
