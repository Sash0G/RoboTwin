# Replay a list of seeds through RoboTwin's scene builder and write one fingerprint per seed, to diff
# against barrel's `//barrel/components/inference/robots/robotwin:scene_fingerprint` output.
#
# This deliberately does NOT reproduce `eval_policy.py`'s episode loop. That loop derives its seeds from
# which scenes the curobo expert solves, which barrel cannot reproduce (it never runs the expert), and it
# draws the instruction off an unseeded `random` module, so its phrasing is not reproducible between two
# upstream runs either. The seeds come in from barrel's run instead, and the protocol below mirrors
# barrel's `robotwin_env.py` / `instructions.py` step for step.
#
#   python script/replay_seeds.py --task-name place_object_stand --seeds seeds.txt --out upstream.json
import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.append("./")
sys.path.append("./description/utils")

import numpy as np
import yaml

from envs import CONFIGS_PATH
from generate_episode_instructions import generate_episode_descriptions

# Mirrors barrel's `scene_fingerprint.py`; bump both together.
FINGERPRINT_VERSION = 1
# Mirrors barrel's `instructions.INSTRUCTION_POOL_SIZE` (upstream eval's `test_num`).
INSTRUCTION_POOL_SIZE = 100
_PLANNED = {"status": "Success", "position": [None]}


def decouple_planner() -> None:
    """Drop the curobo half of `Robot.set_planner`, as barrel's `_decouple_robotwin_planner` does.

    Nothing here plans a motion, and building a curobo planner per embodiment would both require curobo
    and run third-party code between the seeding and `load_actors`. The two TOPP planners upstream builds
    are kept, so the scene is assembled through exactly the same calls barrel makes.
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


def build_args(task_name: str, task_config: str) -> Dict[str, Any]:
    """`eval_policy.main`'s config assembly, matching barrel's `RoboTwinEnv._assemble_args`."""
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = None
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_types = yaml.safe_load(f)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        cam_cfg = yaml.safe_load(f)
    emb = args["embodiment"]
    hct = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cam_cfg[hct]["h"], cam_cfg[hct]["w"]
    robot_file = emb_types[emb[0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        args["left_embodiment_config"] = args["right_embodiment_config"] = yaml.safe_load(f)
    args["render_freq"] = 0
    args["eval_video_log"] = False
    args["eval_mode"] = True  # held-out `unseen` background textures, as eval_policy sets
    return args


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
    """The instruction, drawn where barrel draws it: off the numpy stream left by the scene build."""
    random.seed(seed)
    descriptions = generate_episode_descriptions(task_name, [info], INSTRUCTION_POOL_SIZE)
    candidates = descriptions[0][instruction_type] if descriptions else []
    return str(np.random.choice(candidates)) if candidates else ""


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (bool, int, float, str)) else str(value)


def scene_fingerprint(task: Any, seed: int, instruction: str) -> Dict[str, Any]:
    obs = task.get_obs()
    head_rgb = np.asarray(obs["observation"]["head_camera"]["rgb"], dtype=np.uint8)
    actors = sorted(
        [
            entity.get_name(),
            np.round(entity.get_pose().p, 6).tolist(),
            np.round(entity.get_pose().q, 6).tolist(),
        ]
        for entity in task.scene.get_all_actors()
        if entity.get_name()
    )
    chosen = {
        name: _scalar(value)
        for name, value in sorted(vars(task).items())
        if (name.startswith("selected_") or name.endswith("_id"))
        and isinstance(value, (bool, int, float, str, np.generic))
    }
    return {
        "version": FINGERPRINT_VERSION,
        "seed": int(seed),
        "instruction": instruction,
        "table_z_bias": round(float(task.table_z_bias), 6),
        "chosen": chosen,
        "actors": actors,
        "head_rgb_md5": hashlib.md5(head_rgb.tobytes()).hexdigest(),
    }


def read_seeds(spec: str) -> List[int]:
    text = open(spec, encoding="utf-8").read() if os.path.isfile(spec) else spec
    return [int(token) for token in text.replace(",", " ").split()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--seeds", required=True, help="comma-separated seeds, or a file of seeds")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = read_seeds(args.seeds)
    out_path = os.path.abspath(args.out)

    decouple_planner()
    task_args = build_args(args.task_name, args.task_config)
    task_cls = getattr(importlib.import_module(f"envs.{args.task_name}"), args.task_name)

    rows = []
    for episode_index, seed in enumerate(seeds):
        # Probe the scene for its parameters, then rebuild it: same seed, same episode index, so the
        # instruction is drawn off the stream the real scene left behind. This is barrel's order.
        probe = task_cls()
        probe.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **task_args)
        info = episode_info(probe)
        probe.close_env()

        task = task_cls()
        task.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **task_args)
        instruction = generate_instruction(args.task_name, info, args.instruction_type, seed)
        rows.append(scene_fingerprint(task, seed, instruction))
        task.close_env()
        print(f"seed {seed}: {rows[-1]['head_rgb_md5']}")

    with open(out_path, "w", encoding="utf-8") as out_file:
        json.dump(rows, out_file, indent=2, sort_keys=True)
    print(f"Wrote {len(rows)} scene fingerprints to {out_path}")


if __name__ == "__main__":
    main()
