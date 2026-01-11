from collections import namedtuple
import numpy as np
import string
import random

GROUND_NAME = 'grey'
BLOCK_WIDTH = 2
BLOCK_HEIGHT = BLOCK_WIDTH
GROUND_Y = 0.0

SUCTION_HEIGHT = 1.
GRASP = -np.array([0, BLOCK_HEIGHT + SUCTION_HEIGHT/2])
CARRY_Y = 3*BLOCK_WIDTH + SUCTION_HEIGHT

MOVE_COST = 10.
COST_PER_DIST = 1.
DISTANCE_PER_TIME = 4.0

def generate_non_overlapping_poses(n_blocks, x_min=-4, x_max=5, min_spacing=0, max_spacing=1):
    poses = []
    current_x = x_min + BLOCK_WIDTH / 2
    for i in range(n_blocks):
        remaining_blocks = n_blocks - i - 1
        space_needed = remaining_blocks * (BLOCK_WIDTH + min_spacing) + BLOCK_WIDTH / 2
        available_space = x_max - current_x
        if available_space < space_needed:
            raise ValueError(f"Non c'è abbastanza spazio per {n_blocks} blocchi in [{x_min}, {x_max}]")
        poses.append(np.array([current_x, 0.0]))
        if i < n_blocks - 1:
            spacing = np.random.uniform(min_spacing, max_spacing)
            current_x += BLOCK_WIDTH + spacing
    return poses

def get_block_box(b, p=np.zeros(2)):
    lower = np.array([p[0] - BLOCK_WIDTH/2, p[1]])
    upper = np.array([p[0] + BLOCK_WIDTH/2, p[1] + BLOCK_HEIGHT])
    return lower, upper

def boxes_overlap(box1, box2, tol=0.0):
    lower1, upper1 = box1
    lower2, upper2 = box2
    x_overlap = (lower1[0] < upper2[0] - tol) and (lower2[0] < upper1[0] - tol)
    y_overlap = (lower1[1] < upper2[1] - tol) and (lower2[1] < upper1[1] - tol)
    return x_overlap and y_overlap

def get_block_interval(b, p):
    lower, upper = get_block_box(b, p)
    return lower[0], upper[0]

def interval_contains(i1, i2):
    return (i1[0] <= i2[0]) and (i2[1] <= i1[1])

def collision_test(b1, p1, b2, p2):
    if b1 == b2:
        return False
    return boxes_overlap(
        get_block_box(b1, p1),
        get_block_box(b2, p2),
        tol=0.01
    )

def distance_fn(q1, q2):
    return MOVE_COST + COST_PER_DIST * np.linalg.norm(q2 - q1, ord=1)

def duration_fn(traj):
    distance = sum(np.linalg.norm(q2 - q1) for q1, q2 in zip(traj, traj[1:]))
    return distance / DISTANCE_PER_TIME

def forward_kin(q, g):
    return q + g

def inverse_kin(p, g):
    return p - g

def inverse_kin_fn(b, p, g):
    q = inverse_kin(p, g)
    return (q,)

def get_stack_gen(x_radius=0.5, y_eps=0.02, method='uniform'):
    def gen_fn(b_above, b_below, p_below):
        if not (-0.1 <= p_below[1] <= 0.1):
            return
        cx = float(p_below[0])
        target_y = float(BLOCK_HEIGHT)
        while True:
            if method == 'uniform':
                x = np.random.uniform(cx - x_radius, cx + x_radius)
            elif method == 'normal':
                x = np.random.normal(cx, x_radius/2.0)
            else:
                x = cx
            y = target_y + np.random.uniform(-y_eps, y_eps)
            p_above = np.array([x, y])
            yield (p_above,)
    return gen_fn

def get_region_test(regions):
    def test(b, p, r):
        if r not in regions:
            return False
        x_interval = get_block_interval(b, p)
        region_interval = regions[r]
        contained = interval_contains(region_interval, x_interval)
        if r == YELLOW_NAME:
            if abs(p[1] - PLATFORM_TOP) > 0.1:
                return False
        elif r in [GROUND_NAME, GOAL_NAME]:
            valid_ground = abs(p[1] - 0.0) < 0.1
            valid_stacked = abs(p[1] - BLOCK_HEIGHT) < 0.1
            if not (valid_ground or valid_stacked):
                return False
        return contained
    return test

def get_on_ground_test():
    def test(b, p):
        return abs(p[1] - 0.0) < 0.1
    return test

def plan_motion(q1, q2, fluents):
    """
    Pianificatore di movimento 2D con collision avoidance.
    Considera blocchi statici E la piattaforma gialla come ostacoli.
    Ritorna una tupla (traiettoria,) o None.
    """
    q1, q2 = np.array(q1, dtype=float), np.array(q2, dtype=float)
    if np.allclose(q1, q2):
        return ([q1.copy()],)

    carried_block, grasp = None, None
    placed_blocks = {}
    
    for fluent in fluents:
        if not isinstance(fluent, (list, tuple)) or len(fluent) < 2:
            continue
        name = fluent[0].lower() if isinstance(fluent[0], str) else fluent[0]
        if name == 'atgrasp':
            if len(fluent) >= 4:
                carried_block = fluent[2]
                grasp = np.array(fluent[3], dtype=float)
        elif name == 'atpose':
            if len(fluent) >= 3:
                placed_blocks[fluent[1]] = np.array(fluent[2], dtype=float)
    
    if carried_block and carried_block in placed_blocks:
        del placed_blocks[carried_block]

    PLATFORM_X_MIN = -10.0
    PLATFORM_X_MAX = -5.0
    PLATFORM_Y_MIN = 2.5
    PLATFORM_Y_MAX = 3.5
    
    ROBOT_WIDTH = 1.5  
    ROBOT_HEIGHT = SUCTION_HEIGHT  

    is_carrying = (carried_block is not None) and (grasp is not None)

    def collides_at(robot_conf):
        
        robot_lower = np.array([robot_conf[0] - ROBOT_WIDTH/2, robot_conf[1] - ROBOT_HEIGHT/2])
        robot_upper = np.array([robot_conf[0] + ROBOT_WIDTH/2, robot_conf[1] + ROBOT_HEIGHT/2])
        
        for obs_pos in placed_blocks.values():
            obs_lower = np.array([obs_pos[0] - BLOCK_WIDTH/2, obs_pos[1]])
            obs_upper = np.array([obs_pos[0] + BLOCK_WIDTH/2, obs_pos[1] + BLOCK_HEIGHT])
            
            if (robot_lower[0] < obs_upper[0] - 0.01 and 
                obs_lower[0] < robot_upper[0] - 0.01 and
                robot_lower[1] < obs_upper[1] - 0.01 and 
                obs_lower[1] < robot_upper[1] - 0.01):
                return True
        
        robot_overlaps_platform_x = (robot_lower[0] < PLATFORM_X_MAX - 0.01 and 
                                      PLATFORM_X_MIN + 0.01 < robot_upper[0])
        if robot_overlaps_platform_x:
            if robot_lower[1] < PLATFORM_Y_MAX and robot_upper[1] > PLATFORM_Y_MIN:
                return True
        
        if is_carrying:
            block_pos = robot_conf + grasp  
            block_lower = np.array([block_pos[0] - BLOCK_WIDTH/2, block_pos[1]])
            block_upper = np.array([block_pos[0] + BLOCK_WIDTH/2, block_pos[1] + BLOCK_HEIGHT])
            
            for obs_pos in placed_blocks.values():
                obs_lower = np.array([obs_pos[0] - BLOCK_WIDTH/2, obs_pos[1]])
                obs_upper = np.array([obs_pos[0] + BLOCK_WIDTH/2, obs_pos[1] + BLOCK_HEIGHT])
                
                if (block_lower[0] < obs_upper[0] - 0.01 and 
                    obs_lower[0] < block_upper[0] - 0.01 and
                    block_lower[1] < obs_upper[1] - 0.01 and 
                    obs_lower[1] < block_upper[1] - 0.01):
                    return True
            
            block_overlaps_platform_x = (block_lower[0] < PLATFORM_X_MAX - 0.01 and 
                                          PLATFORM_X_MIN + 0.01 < block_upper[0])
            
            if block_overlaps_platform_x:
                if block_lower[1] >= PLATFORM_Y_MAX - 0.01:
                    pass  
                elif block_lower[1] < PLATFORM_Y_MAX and block_upper[1] > PLATFORM_Y_MIN:
                    return True
            
        return False

    def path_collides(path):
        for i in range(len(path) - 1):
            p1, p2 = np.array(path[i]), np.array(path[i+1])
            length = np.linalg.norm(p2 - p1)
            steps = max(2, int(length / 0.05))
            for t in range(steps + 1):
                conf = p1 + (p2 - p1) * (t / steps)
                if collides_at(conf):
                    return True
        return False

    if not path_collides([q1, q2]):
        return ([q1.copy(), q2.copy()],)

    max_obs_top = PLATFORM_Y_MAX  
    if placed_blocks:
        block_tops = max(p[1] + BLOCK_HEIGHT for p in placed_blocks.values())
        max_obs_top = max(max_obs_top, block_tops)
    
    if is_carrying:
        safe_robot_y = max_obs_top - grasp[1] + 0.1
    else:
        safe_robot_y = max_obs_top + ROBOT_HEIGHT/2 + 0.1

    max_iter = 2000
    step = 0.5
    xs = [q1[0], q2[0]] + [p[0] for p in placed_blocks.values()]
    bounds = ([min(xs) - 2, GROUND_Y], [max(xs) + 2, safe_robot_y + 1])
    
    nodes = [q1.copy()]
    parents = {0: None}
    
    for _ in range(max_iter):
        if random.random() < 0.15:
            sample = q2.copy()
        else:
            sample = np.array([random.uniform(bounds[0][0], bounds[1][0]),
                              random.uniform(bounds[0][1], bounds[1][1])])
        
        dists = [np.linalg.norm(sample - n) for n in nodes]
        near_idx = np.argmin(dists)
        near = nodes[near_idx]
        
        diff = sample - near
        dist = np.linalg.norm(diff)
        new_pt = sample if dist <= step else near + diff / dist * step
        
        if path_collides([near, new_pt]):
            continue
        
        new_idx = len(nodes)
        nodes.append(new_pt)
        parents[new_idx] = near_idx
        
        if np.linalg.norm(new_pt - q2) <= step and not path_collides([new_pt, q2]):
            path = [q2.copy()]
            idx = new_idx
            while idx is not None:
                path.append(nodes[idx])
                idx = parents.get(idx)
            path.reverse()
            return (path,)
    
    return None



TAMPState = namedtuple('TAMPState', ['robot_confs', 'holding', 'block_poses'])
TAMPProblem = namedtuple('TAMPProblem', ['initial', 'regions',
                                         'goal_conf', 'goal_regions',
                                         'assignments', 'goal_on'])

GOAL_NAME = 'red'
YELLOW_NAME = 'yellow'
INITIAL_CONF = np.array([-5, CARRY_Y + 1])

REGIONS = {
    GROUND_NAME: (-10, 10),
    GOAL_NAME: (5, 10),
    YELLOW_NAME: (-10, -5),
}

ENVIRONMENT_NAMES = [GROUND_NAME]

def make_blocks(num):
    return [string.ascii_uppercase[i] for i in range(num)]

PLATFORM_TOP = 3.5  

def get_on_platform_test():
    def test(b, p):

        return abs(p[1] - PLATFORM_TOP) < 0.1
    return test

def get_pose_gen(regions):
    def gen_fn(block, region):
        x1, x2 = regions[region]
        if region == YELLOW_NAME:
            y = PLATFORM_TOP  
        else:
            y = 0.0  
        for _ in range(100):
            x = np.random.uniform(x1, x2)
            p = np.array([x, y])
            yield (p,)
    return gen_fn

def new_problem(n_blocks=2, n_goals=2, n_robots=1):
    confs = [INITIAL_CONF, np.array([-1, 1]) * INITIAL_CONF]
    robots = ['r{}'.format(x) for x in range(n_robots)]
    initial_confs = dict(zip(robots, confs))
    poses = generate_non_overlapping_poses(n_blocks, x_min=-4, x_max=5,
                                           min_spacing=0, max_spacing=1)
    blocks = make_blocks(len(poses))
    initial = TAMPState(initial_confs, {}, dict(zip(blocks, poses)))
    goal_regions = {}
    available_regions = [YELLOW_NAME, GOAL_NAME]
    for i, block in enumerate(blocks[:n_goals]):
        region_index = i % len(available_regions)
        goal_regions[block] = available_regions[region_index]
    goal_on = []
    goal_conf = initial_confs.copy()
    return TAMPProblem(initial, REGIONS, goal_conf, goal_regions, assignments=None, goal_on=goal_on)

def tight(n_blocks=2, n_goals=2, n_robots=1):
    confs = [INITIAL_CONF, np.array([-1, 1]) * INITIAL_CONF]
    robots = ['r{}'.format(x) for x in range(n_robots)]
    initial_confs = dict(zip(robots, confs))
    poses = generate_non_overlapping_poses(n_blocks, x_min=-4, x_max=5,
                                           min_spacing=0, max_spacing=1)
    blocks = make_blocks(len(poses))
    initial = TAMPState(initial_confs, {}, dict(zip(blocks, poses)))
    goal_regions = {block: GOAL_NAME for block in blocks[:n_goals]}
    if n_blocks >= 2:
        goal_on = [(blocks[1], blocks[0])]
    else:
        goal_on = []
    goal_conf = initial_confs.copy()
    return TAMPProblem(initial, REGIONS, goal_conf, goal_regions,
                       assignments=None, goal_on=goal_on)



PROBLEMS = [tight, new_problem]

def draw_robot(viewer, robot, pose, **kwargs):
    x, y = pose
    viewer.draw_robot(x, y, name=robot, **kwargs)

def draw_block(viewer, block, pose, **kwargs):
    x, y = pose
    viewer.draw_block(x, y, BLOCK_WIDTH, BLOCK_HEIGHT, name=block, **kwargs)

def draw_state(viewer, state, colors):
    viewer.clear_state()
    print(state)
    for robot, conf in state.robot_confs.items():
        draw_robot(viewer, robot, conf)
    for block, pose in state.block_poses.items():
        draw_block(viewer, block, pose, color=colors[block])
    for robot, holding in state.holding.items():
        block, grasp = holding
        pose = forward_kin(state.robot_confs[robot], grasp)
        draw_block(viewer, block, pose, color=colors[block])
    viewer.tk.update()

def get_random_seed():
    return np.random.get_state()[1][0]



def apply_action(state, action):
    robot_confs, holding, block_poses = state
    name, args = action[:2]
    if name == 'move':
        if len(args) == 4:
            robot, _, traj, _ = args
        else:
            robot, q1, q2 = args
            traj = [q1, q2]
        for conf in traj[1:]:
            if holding.get(robot):
                block, grasp = holding[robot]
                p = forward_kin(conf, grasp)
                box_moving = get_block_box(block, p)
                for b2, p2 in block_poses.items():
                    if boxes_overlap(box_moving, get_block_box(b2, p2), tol=0.02):
                        raise RuntimeError(f'Collision detected during apply_action move: {robot} holding {block} collided with {b2} at conf {conf}')
        robot_confs[robot] = conf
        yield TAMPState(robot_confs, holding, block_poses)

    elif name == 'pick':
        robot, block, _, grasp, _ = args
        holding[robot] = (block, grasp)
        del block_poses[block]
        yield TAMPState(robot_confs, holding, block_poses)
    elif name in ['place', 'place-ground', 'place-platform']:
        robot, block, pose, _, _ = args
        del holding[robot]
        block_poses[block] = pose
        yield TAMPState(robot_confs, holding, block_poses)
    elif name == 'stack':
        if len(args) == 7:
            robot, block, pose, _, _, _, _ = args
        else:
            robot, block, pose, _, _, _ = args
        del holding[robot]
        block_poses[block] = pose
        yield TAMPState(robot_confs, holding, block_poses)
    else:
        raise ValueError(name)



def prune_duplicates(traj):
    new_traj = [traj[0]]
    for conf in traj[1:]:
        if 0 < np.linalg.norm(np.array(conf) - np.array(new_traj[-1])):
            new_traj.append(conf)
    return new_traj

def get_value_at_time(traj, fraction):
    waypoints = prune_duplicates(traj)
    if len(waypoints) == 1:
        return waypoints[0]
    distances = [0.] + [np.linalg.norm(np.array(q2) - np.array(q1))
                        for q1, q2 in zip(waypoints, waypoints[1:])]
    cum_distances = np.cumsum(distances)
    cum_fractions = np.minimum(cum_distances / cum_distances[-1],
                               np.ones(cum_distances.shape))
    index = np.digitize(fraction, cum_fractions, right=False)
    if index == len(waypoints):
        index -= 1
    waypoint_fraction = ((fraction - cum_fractions[index - 1]) /
                          (cum_fractions[index] - cum_fractions[index - 1]))
    waypoint1, waypoint2 = np.array(waypoints[index - 1]), np.array(waypoints[index])
    conf = (1 - waypoint_fraction) * waypoint1 + waypoint_fraction * waypoint2
    return conf

def update_state(state, action, t):
    robot_confs, holding, block_poses = state
    name, args, start, duration = action
    fraction = max(0, min(float(t) / duration, 1))
    threshold = 0.5
    if name == 'move':
        robot, _, traj, _ = args
        robot_confs[robot] = get_value_at_time(traj, fraction)
    elif name == 'pick':
        robot, block, pose, grasp, conf = args[:5]
        grasp_pos = pose - grasp
        traj_down = [conf, grasp_pos]
        traj_up = [grasp_pos, conf]
        if fraction < threshold:
            robot_confs[robot] = get_value_at_time(traj_down, fraction / threshold)
        else:
            if not holding.get(robot):
                holding[robot] = (block, grasp)
                block_poses.pop(block, None)
            robot_confs[robot] = get_value_at_time(traj_up,
                                                   (fraction - threshold) / (1 - threshold))
    elif name in ['place', 'place-ground', 'place-platform']:
        robot, block, pose, grasp, conf = args[:5]
        place_pos = pose - grasp
        traj_down = [conf, place_pos]
        traj_up = [place_pos, conf]
        if fraction < threshold:
            robot_confs[robot] = get_value_at_time(traj_down, fraction / threshold)
        else:
            if holding.get(robot):
                holding.pop(robot, None)
                block_poses[block] = pose
            robot_confs[robot] = get_value_at_time(traj_up,
                                                   (fraction - threshold) / (1 - threshold))
    elif name == 'stack':
        if len(args) >= 5:
            robot, block, pose, grasp, conf = args[:5]
            place_pos = pose - grasp
            traj_down = [conf, place_pos]
            traj_up = [place_pos, conf]
            if fraction < threshold:
                robot_confs[robot] = get_value_at_time(traj_down, fraction / threshold)
            else:
                if holding.get(robot):
                    holding.pop(robot, None)
                    block_poses[block] = pose
                robot_confs[robot] = get_value_at_time(traj_up,
                                                       (fraction - threshold) / (1 - threshold))
    return TAMPState(robot_confs, holding, block_poses)
