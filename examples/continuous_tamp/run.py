from __future__ import print_function

import os
import numpy as np
import time

from pddlstream.algorithms.meta import solve, create_parser
from examples.continuous_tamp.primitives import (
    get_pose_gen, collision_test, distance_fn, inverse_kin_fn,
    plan_motion, PROBLEMS, draw_state, get_random_seed,
    SUCTION_HEIGHT, MOVE_COST, GRASP, update_state, ENVIRONMENT_NAMES,
    YELLOW_NAME, get_stack_gen, get_on_ground_test, get_on_platform_test,
    BLOCK_HEIGHT, get_block_interval, interval_contains,
)
from pddlstream.algorithms.downward import get_cost_scale
from pddlstream.algorithms.constraints import PlanConstraints
from pddlstream.algorithms.visualization import VISUALIZATIONS_DIR
from pddlstream.language.external import defer_shared, get_defer_all_unbound, get_defer_any_unbound
from pddlstream.language.constants import And, Equal, PDDLProblem, TOTAL_COST, print_solution
from pddlstream.language.function import FunctionInfo
from pddlstream.language.generator import from_gen_fn, from_test, from_fn
from pddlstream.language.stream import StreamInfo
from pddlstream.language.temporal import get_end, compute_duration, retime_plan
from pddlstream.utils import (
    ensure_dir, safe_rm_dir, user_input, read, INF, get_file_path,
    str_from_object, sorted_str_from_list, implies, inclusive_range, Profiler
)
from examples.continuous_tamp.viewer import ContinuousTMPViewer
from examples.discrete_tamp.viewer import COLORS

def create_problem(tamp_problem, hand_empty=False, manipulate_cost=1.0):
    initial = tamp_problem.initial
    assert not initial.holding

    init = [
        Equal(('Cost',), manipulate_cost),
        Equal((TOTAL_COST,), 0),
    ] + [
        ('Region', r) for r in tamp_problem.regions
    ] + [
        ('Placeable', b, r)
        for b in initial.block_poses.keys()
        for r in tamp_problem.regions
        if (r in ENVIRONMENT_NAMES) or (r in tamp_problem.goal_regions.values())
    ]

    if tamp_problem.assignments:
        for robot, blocks in tamp_problem.assignments.items():
            for block in blocks:
                init.append(('Assigned', robot, block))
    else:
        for robot in initial.robot_confs.keys():
            for block in initial.block_poses.keys():
                init.append(('Assigned', robot, block))

    goal_literals = []

    for b, p in initial.block_poses.items():
        init += [
            ('Block', b),
            ('Pose', b, p),
            ('AtPose', b, p),
            ('Clear', b),
        ]
        for r_name, (x1, x2) in tamp_problem.regions.items():
            x_interval = get_block_interval(b, p)

            if not interval_contains((x1, x2), x_interval):
                continue

            y = p[1]

            if r_name == YELLOW_NAME:
                expected_y = 2.5 + BLOCK_HEIGHT / 2
                if abs(y - expected_y) < 0.1:
                    init.append(('In', b, r_name))
            else:
                valid_ground = abs(y - 0.0) < 0.1
                valid_stacked = abs(y - BLOCK_HEIGHT) < 0.1

                if valid_ground or valid_stacked:
                    init.append(('In', b, r_name))

    elevated_blocks = [b for b, r in tamp_problem.goal_regions.items()
                       if r == YELLOW_NAME]
    ground_blocks = [b for b, r in tamp_problem.goal_regions.items()
                     if r != YELLOW_NAME]

    for b in elevated_blocks:
        r = tamp_problem.goal_regions[b]
        goal_literals.append(('In', b, r))
    for b in ground_blocks:
        r = tamp_problem.goal_regions[b]
        goal_literals.append(('In', b, r))

    if hasattr(tamp_problem, 'goal_on') and tamp_problem.goal_on:
        stacked_blocks = set()
        base_blocks = set()

        for b_above, b_below in tamp_problem.goal_on:
            stacked_blocks.add(b_above)
            base_blocks.add(b_below)
            goal_literals.append(('On', b_above, b_below))

    robots_list = list(initial.robot_confs.items())

    for r, q in robots_list:
        init += [
            ('Robot', r),
            ('CanMove', r),
            ('Conf', q),
            ('AtConf', r, q),
            ('HandEmpty', r),
        ]
        if hand_empty:
            goal_literals.append(('HandEmpty', r))

        if tamp_problem.goal_conf is not None:
            if isinstance(tamp_problem.goal_conf, dict):
                if r in tamp_problem.goal_conf:
                    goal_literals.append(('AtConf', r, tamp_problem.goal_conf[r]))
            else:
                goal_literals.append(('AtConf', r, tamp_problem.goal_conf))

    goal = And(*goal_literals)

    return init, goal

def pddlstream_from_tamp(tamp_problem, use_stream=True, collisions=True):
    domain_pddl = read(get_file_path(__file__, 'domain.pddl'))
    external_paths = []
    if use_stream:
        external_paths.append(get_file_path(__file__, 'stream.pddl'))
    external_pddl = [read(path) for path in external_paths]

    constant_map = {}

    def region_gen_wrapper(block, region):
        gen = get_pose_gen(tamp_problem.regions)(block, region)
        for (p,) in gen:
            yield (p,)

    stream_map = {
        's-grasp': from_fn(lambda b: (GRASP,)),
        's-region': from_gen_fn(region_gen_wrapper),
        's-stack': from_gen_fn(get_stack_gen()),
        's-ik': from_fn(inverse_kin_fn),
        's-motion': from_fn(plan_motion),
        't-on-ground': from_test(get_on_ground_test()),
        't-on-platform': from_test(get_on_platform_test()),
        't-cfree': from_test(lambda *args: implies(collisions, not collision_test(*args))),
        'dist': distance_fn,
    }

    init, goal = create_problem(tamp_problem)

    return PDDLProblem(domain_pddl, constant_map, external_pddl, stream_map, init, goal)

def display_plan(tamp_problem, plan, display=True, save=False, time_step=0.025, sec_per_step=1e-3):
    if save:
        example_name = 'continuous_tamp'
        directory = os.path.join(VISUALIZATIONS_DIR, '{}/'.format(example_name))
        safe_rm_dir(directory)
        ensure_dir(directory)
    colors = dict(zip(sorted(tamp_problem.initial.block_poses.keys()), COLORS))
    show_platform = YELLOW_NAME in tamp_problem.goal_regions.values()

    viewer = ContinuousTMPViewer(
        SUCTION_HEIGHT,
        tamp_problem.regions,
        title='Continuous TAMP',
        show_platform=show_platform
    )
    state = tamp_problem.initial

    print()
    print(state)
    duration = compute_duration(plan)
    real_time = (None if sec_per_step is None
                else (duration * sec_per_step / time_step))
    print('Duration: {} | Step size: {} | Real time: {}'.format(
        duration, time_step, real_time))

    draw_state(viewer, state, colors)
    if display:
        user_input('Start?')

    if plan is not None:
        for t in inclusive_range(0, duration, time_step):
            for action in plan:
                if action.start <= t <= get_end(action):
                    update_state(state, action, t - action.start)
            print('t={} | {}'.format(t, state))
            draw_state(viewer, state, colors)

            if save:
                viewer.save(os.path.join(directory, 't={}'.format(t)))
            if display:
                if sec_per_step is None:
                    user_input('Continue?')
                else:
                    time.sleep(sec_per_step)

    if display:
        user_input('Finish?')
    return state

def initialize(parser):
    parser.add_argument('-c', '--cfree', action='store_true',
                       help='Disables collisions')
    parser.add_argument('-t', '--max_time', default=30, type=int,
                       help='The max time')
    parser.add_argument('-n', '--number', default=2, type=int,
                       help='The number of blocks')
    parser.add_argument('-p', '--problem', default='tight',
                       help='The name of the problem to solve')
    parser.add_argument('-v', '--visualize', action='store_true',
                       help='Visualizes graphs')

    args = parser.parse_args()
    print('Arguments:', args)
    np.set_printoptions(precision=2)
    print('Random seed:', get_random_seed())

    problem_from_name = {fn.__name__: fn for fn in PROBLEMS}
    if args.problem not in problem_from_name:
        raise ValueError(args.problem)

    print('Problem:', args.problem)
    problem_fn = problem_from_name[args.problem]
    tamp_problem = problem_fn(args.number)
    print(tamp_problem)

    return tamp_problem, args

def dump_pddlstream(pddlstream_problem):
    print('Initial:', sorted_str_from_list(pddlstream_problem.init))
    print('Goal:', str_from_object(pddlstream_problem.goal))

def main():
    parser = create_parser()
    tamp_problem, args = initialize(parser)

    stream_info = {
        's-region': StreamInfo(defer_fn=defer_shared),
        's-grasp': StreamInfo(defer_fn=defer_shared),
        's-stack': StreamInfo(defer_fn=defer_shared, eager=False),
        's-ik': StreamInfo(defer_fn=get_defer_all_unbound(inputs='?g')),
        's-motion': StreamInfo(defer_fn=get_defer_any_unbound()),
        't-cfree': StreamInfo(defer_fn=get_defer_any_unbound(), eager=False, verbose=False),
        't-on-ground': StreamInfo(eager=True, negate=False),
        't-on-platform': StreamInfo(eager=True, negate=False),
        'dist': FunctionInfo(eager=False, defer_fn=get_defer_any_unbound(),
                            opt_fn=lambda q1, q2: MOVE_COST),
    }

    constraints = PlanConstraints(
        skeletons=None,
        exact=True,
        max_cost=INF
    )

    pddlstream_problem = pddlstream_from_tamp(
        tamp_problem,
        collisions=not args.cfree,
        use_stream=True
    )
    dump_pddlstream(pddlstream_problem)

    planner = 'max-astar'
    effort_weight = 1. / get_cost_scale()

    print('\nSolving...')
    with Profiler(field='cumtime', num=20):
        solution = solve(
            pddlstream_problem,
            algorithm=args.algorithm,
            constraints=constraints,
            stream_info=stream_info,
            replan_actions=set(),
            planner=planner,
            max_planner_time=10,
            max_time=args.max_time,
            max_iterations=INF,
            debug=False,
            verbose=True,
            unit_costs=args.unit,
            success_cost=INF,
            unit_efforts=True,
            effort_weight=effort_weight,
            search_sample_ratio=1,
            visualize=args.visualize
        )

    print_solution(solution)
    plan, cost, evaluations = solution

    if plan is not None:
        print('\n✓ Plan found! Cost:', cost)
        display_plan(tamp_problem, retime_plan(plan))
    else:
        print('\n✗ No plan found')

if __name__ == '__main__':
    main()
