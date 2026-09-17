# Evaluate a policy on an EXACT list of seeds, one episode per seed, instead of deriving the seeds the
# way `eval_policy.py` does.
#
# `eval_policy.py` starts at `100000 * (1 + seed)` and then walks forward, keeping only the seeds whose
# scene the curobo expert solves. That list cannot be reproduced anywhere the expert does not run, so it
# cannot be lined up with a barrel eval. Here the seeds come in from a file, every one of them is used,
# and the scene build, the instruction draw and `eval_mode` follow barrel's `robotwin_env.py` /
# `instructions.py` step for step, so the same seed gives the same scene and the same instruction.
#
#   python script/eval_policy_seeds.py --config policy/pi05/deploy_policy.yml --overrides \
#       --task_name place_a2b_right --task_config demo_randomized --policy_name pi05 \
#       --seed_list /workspace/seeds.txt ...
import argparse
import importlib
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

import numpy as np
import yaml

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

# Mirrors barrel's `instructions.INSTRUCTION_POOL_SIZE`.
INSTRUCTION_POOL_SIZE = 100
_PLANNED = {"status": "Success", "position": [None]}

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)()
    except Exception:
        raise SystemExit("No Task")


def eval_function_decorator(policy_name, model_name):
    policy_model = importlib.import_module(policy_name)
    return getattr(policy_model, model_name)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def read_seeds(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        seeds = [int(token) for token in f.read().replace(",", " ").split()]
    if not seeds:
        raise SystemExit(f"No seeds in {path}")
    return seeds


def decouple_planner() -> None:
    """Drop the curobo half of `Robot.set_planner`, as barrel's `_decouple_robotwin_planner` does.

    Nothing here plans a motion (the policy owns IK and drives joints), so building a curobo planner per
    embodiment would only run third-party code between the seeding and `load_actors`, where barrel runs
    none. The two TOPP planners upstream builds under `need_topp` are kept, `take_action` needs them.
    """
    from envs.robot.planner import MplibPlanner
    from envs.robot.robot import Robot

    def _skip_planner_setup(self: Any, scene: Any = None) -> None:
        self.left_planner = self.right_planner = None
        self.communication_flag = False
        self.left_mplib_planner = MplibPlanner(
            self.left_urdf_path,
            self.left_srdf_path,
            self.left_move_group,
            self.left_entity_origion_pose,
            self.left_entity,
            self.left_planner_type,
            scene,
        )
        self.right_mplib_planner = MplibPlanner(
            self.right_urdf_path,
            self.right_srdf_path,
            self.right_move_group,
            self.right_entity_origion_pose,
            self.right_entity,
            self.right_planner_type,
            scene,
        )

    Robot.communication_flag = False
    Robot.left_planner = None
    Robot.right_planner = None
    Robot.set_planner = _skip_planner_setup


def stub_planning(task: Any) -> None:
    """Let the expert script run without a motion planner, on a scene to be thrown away."""
    task.move = lambda *args, **kwargs: True
    for arm in ("left", "right"):
        setattr(task.robot, f"{arm}_plan_path", lambda *args, **kwargs: _PLANNED)
        setattr(
            task.robot,
            f"{arm}_plan_multi_path",
            lambda targets, *args, **kwargs: {
                "status": ["Success"] * len(targets),
                "position": [[None]] * len(targets),
            },
        )


def episode_info(task: Any) -> Dict[str, str]:
    stub_planning(task)
    try:
        info = task.play_once() or {}
    except Exception as error:
        print(f"expert script yielded no episode info: {error}")
        return {}
    return dict(info.get("info", {}))


def generate_instruction(task_name: str, info: Dict[str, str], instruction_type: str, seed: int) -> str:
    """Drawn where barrel draws it: off the numpy stream the scene build left behind."""
    random.seed(seed)
    descriptions = generate_episode_descriptions(task_name, [info], INSTRUCTION_POOL_SIZE)
    candidates = descriptions[0][instruction_type] if descriptions else []
    return str(np.random.choice(candidates)) if candidates else ""


def eval_policy_on_seeds(task_name, TASK_ENV, args, model, seeds, video_size, instruction_type):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    eval_func = eval_function_decorator(args["policy_name"], "eval")
    reset_func = eval_function_decorator(args["policy_name"], "reset_model")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]
    results = []

    for episode_index, seed in enumerate(seeds):
        # Probe the scene for its parameters, then rebuild it, so the instruction is drawn off the
        # stream the real scene left behind. This is barrel's order.
        try:
            probe = TASK_ENV.__class__()
            probe.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **args)
        except UnStableError as error:
            print(f"\033[91mseed {seed} never settles, recording it as a failure: {error}\033[0m")
            results.append({"seed": seed, "success": False, "instruction": "", "note": "unstable"})
            TASK_ENV.test_num += 1
            continue
        info = episode_info(probe)
        probe.close_env()

        TASK_ENV.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **args)
        instruction = generate_instruction(task_name, info, instruction_type, seed)
        TASK_ENV.set_instruction(instruction=instruction)
        print(f"\033[36mseed {seed}: {instruction}\033[0m")

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", video_size,
                    "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_seed{seed}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        TASK_ENV.suc += int(succ)
        TASK_ENV.test_num += 1
        results.append({"seed": seed, "success": succ, "instruction": instruction})
        print("\033[92mSuccess!\033[0m" if succ else "\033[91mFail!\033[0m")

        TASK_ENV.close_env(clear_cache=((episode_index + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m, seed: \033[90m{seed}\033[0m\n"
        )

    return results


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    seeds = read_seeds(usr_args["seed_list"])

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise SystemExit("No embodiment files")
        return robot_file

    def get_embodiment_config(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args["eval_video_log"]:
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        args["eval_video_save_dir"] = save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    print("\033[95mSeeds:\033[0m " + ", ".join(str(seed) for seed in seeds))
    print("\n==================================")

    decouple_planner()
    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    model = get_model(usr_args)
    results = eval_policy_on_seeds(
        task_name, TASK_ENV, args, model, seeds, video_size, instruction_type
    )

    with open(os.path.join(save_dir, "_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_time}\n\nInstruction Type: {instruction_type}\n\n")
        for row in results:
            f.write(f"{row['seed']}\t{row['success']}\t{row.get('instruction', '')}\n")
        f.write(f"\nSuccess rate: {sum(r['success'] for r in results)}/{len(results)}\n")
    print(f"Data has been saved to {save_dir}/_result.txt")


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        config.update(parse_override_pairs(args.overrides))
    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    main(parse_args_and_config())
