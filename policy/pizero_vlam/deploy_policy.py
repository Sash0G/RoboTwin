"""RoboTwin eval adapter for the barrel PiZero-FM / Qwen3-VL bimanual VLAM.

RoboTwin's ``script/eval_policy.py`` owns the episode loop (seeding, expert check, ``step_lim``,
success checking, video). This module only turns one RoboTwin observation into one action chunk:

    obs joints[14] --FK--> per-arm EEF pose --VLAM--> per-arm EEF deltas --IK--> qpos[14]

The FK/IK chain is barrel's ``PiperArmKinematics``, which mirrors the dataset converter
(``barrel/components/data/postprocess/datasets/raw/robotwin_dataset.py``) exactly: same Piper URDF,
same ``link6`` end-effector, one 6-joint serial chain run once per arm. Keeping the chain identical is
what makes eval frame-consistent with training, and is why we go through joint space rather than
RoboTwin's ``action_type='ee'`` (which expects a pose in the simulator's own frame).

DELTA ANCHORING -- the training control frames are ``ROBOT_BASE_DELTA`` (translation) and ``EEF_DELTA``
(rotation), and ``RoboticsDataset`` converts *every* future control point relative to the observation
pose at a single ``sequence_base_indices`` snapshot -- not step-to-step. So each of the 5 control points
is a displacement measured from the pose observed at the moment of the query, the last one being the
full 1.0 s displacement (hence the widened +/-[0.12, 0.08, 0.10] m ``translation_control_norm`` bounds
in the training config). All 5 deltas are therefore applied to a single pose snapshot taken before the
chunk starts executing. Re-reading the
EEF pose on each step and applying that step's delta to it would compound the deltas and overshoot.

The current joints *are* re-read every step, but only to warm-start IK -- that keeps successive
solutions on the same IK branch without affecting the target pose.
"""
import os
import sys

import numpy as np

# Model camera key <- RoboTwin observation camera key. 'wrist' is the left arm's view, matching the
# dataset transform (cam_high -> main, cam_left_wrist -> wrist, cam_right_wrist -> wrist_right).
CAMERA_MAP = {
    "head_camera": "main",
    "left_camera": "wrist",
    "right_camera": "wrist_right",
}

# The dataset converter does NOT store the raw FK rotation: `_split_bimanual_ee_pose` runs every
# per-arm rotation through `rotation_eef_to_base_frame` (barrel .../lerobot/df_utils/df_transform.py),
# which right-multiplies by this matrix -- the canonical-Franka-EEF -> ROBOT_BASE convention -- before
# quaternizing. So the pose space the model was trained in is A = R_fk @ M, not R_fk. Both directions
# have to go through it: the proprio rotation we feed in, and the delta we get back (the rotation delta
# is EEF-relative, so it is M-conjugated: DR_model = M @ DR_fk @ M). M is its own inverse.
EEF_TO_BASE_ROTATION = np.diag([1.0, -1.0, -1.0])


def _ensure_barrel_importable(barrel_root: str) -> str:
    """Put the barrel repo root on sys.path.

    The exported model in ``hf_export/<session>/src/`` is self-contained apart from
    ``barrel.core.databib`` (a pure config/dataclass library), so this pulls in nothing that
    conflicts with RoboTwin's sapien/mplib stack.
    """
    barrel_root = os.path.abspath(os.path.expanduser(barrel_root))
    if not os.path.isdir(os.path.join(barrel_root, "barrel")):
        raise FileNotFoundError(
            f"barrel_root={barrel_root!r} does not look like a barrel checkout "
            f"(no 'barrel/' package inside it)"
        )
    if barrel_root not in sys.path:
        sys.path.insert(0, barrel_root)
    return barrel_root


class PiZeroVLAMPolicy:
    """Stateless-per-chunk bimanual policy: RoboTwin observation -> 14-dim qpos targets."""

    def __init__(
        self,
        ckpt_path: str,
        barrel_root: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        urdf_path: str = "",
        ee_link_name: str = "",
        dataset_name: str = "robotwin",
        execute_steps: int = 5,
    ) -> None:
        _ensure_barrel_importable(barrel_root)

        import torch

        from barrel.components.geometry.rotation_transforms import RotationFormat, convert_rotation
        from barrel.components.inference.api.core.action import EndEffectorPoseAction
        from barrel.components.inference.models.vlams.vlam_inference import (
            RobotObservation as VLAMRobotObservation,
            VLAMInference,
            VLAMInferenceConfig,
        )
        from barrel.components.inference.robots.robotwin.kinematics import (
            ARM_SLICES,
            DEFAULT_EE_LINK_NAME,
            DEFAULT_URDF_PATH,
            GRIPPER_INDICES,
            NUM_ARM_JOINTS,
            NUM_ARMS,
            PiperArmKinematics,
            pose_from_rotmat_translation,
        )

        self._torch = torch
        self._convert_rotation = convert_rotation
        self._quaternion = RotationFormat.QUATERNION
        self._EndEffectorPoseAction = EndEffectorPoseAction
        self._VLAMRobotObservation = VLAMRobotObservation
        self._pose_from_rotmat_translation = pose_from_rotmat_translation

        self.ARM_SLICES = ARM_SLICES
        self.GRIPPER_INDICES = GRIPPER_INDICES
        self.NUM_ARMS = NUM_ARMS
        self.NUM_ARM_JOINTS = NUM_ARM_JOINTS

        # The cuDNN MHA SDPA kernel fails on the current torch/GPU combo; the memory-efficient / math
        # kernels give identical results (same workaround as barrel's own RoboTwin adapter).
        torch.backends.cuda.enable_cudnn_sdp(False)

        self.inference = VLAMInference(
            VLAMInferenceConfig(ckpt_path=ckpt_path, device=device, dtype=dtype)
        )
        self.processor = self.inference.processor
        self.delta_mode = self.inference.delta_mode
        self.dataset_name = dataset_name

        self._kinematics = PiperArmKinematics(
            urdf_path or DEFAULT_URDF_PATH,
            ee_link_name=ee_link_name or DEFAULT_EE_LINK_NAME,
        )

        # VLAMInference.step reads processor.config.num_arms to split the per-arm control, so a
        # checkpoint exported before bimanual support simply cannot drive this adapter.
        num_arms = getattr(self.processor.config, "num_arms", None)
        if num_arms is None:
            raise ValueError(
                f"The checkpoint's exported processor has no `num_arms`, so it predates bimanual "
                f"support and cannot produce per-arm control: {ckpt_path}"
            )
        if num_arms != NUM_ARMS:
            raise ValueError(
                f"This adapter is bimanual (num_arms={NUM_ARMS}); the checkpoint's processor reports "
                f"num_arms={num_arms}. Wrong checkpoint?"
            )

        self._horizon = self.processor.control_io_config.future_controls_sequence_length
        if execute_steps < 1:
            raise ValueError(f"execute_steps must be >= 1, got {execute_steps}")
        if execute_steps > self._horizon:
            raise ValueError(
                f"execute_steps={execute_steps} exceeds the model's control horizon "
                f"{self._horizon}. Beyond the horizon there is no prediction to execute -- repeating "
                f"the last (full-1.0s) delta would march the arm another full displacement per step."
            )
        self.execute_steps = execute_steps

    # ---------------------------------------------------------------- observation

    def _joints14(self, observation) -> np.ndarray:
        return np.asarray(observation["joint_action"]["vector"], dtype=np.float64)

    def _per_arm_joints(self, observation) -> np.ndarray:
        """`[NUM_ARMS, 6]` arm joints (grippers dropped)."""
        joints14 = self._joints14(observation)
        return np.stack([joints14[self.ARM_SLICES[arm]] for arm in range(self.NUM_ARMS)])

    def _eef_poses(self, observation) -> np.ndarray:
        """Per-arm FK: `[NUM_ARMS, 4, 4]` EEF poses in the arm's own base frame."""
        per_arm = self._per_arm_joints(observation)
        poses = [
            self._kinematics.fk(self._torch.from_numpy(per_arm[arm])).cpu().numpy()
            for arm in range(self.NUM_ARMS)
        ]
        return np.stack(poses).astype(np.float64)

    def _images(self, observation):
        """Resize each RoboTwin view to the size its camera was trained at -> `[H=1, h, w, c]`."""
        cameras = observation["observation"]
        # get_obs() builds this from cameras.get_config(), then only fills in 'rgb' when the task
        # config's data_type enables it -- so a present-but-imageless camera is a distinct failure.
        missing = [env_cam for env_cam in CAMERA_MAP if env_cam not in cameras]
        no_rgb = [env_cam for env_cam in CAMERA_MAP if env_cam in cameras and "rgb" not in cameras[env_cam]]
        if missing or no_rgb:
            raise KeyError(
                f"RoboTwin observation cannot feed this model, which was trained on all of "
                f"{sorted(CAMERA_MAP)}: absent={missing}, present-but-no-rgb={no_rgb}. Enable "
                f"camera.collect_head_camera / camera.collect_wrist_camera (and the rgb data_type) in "
                f"task_config/{{task_config}}.yml."
            )
        return {
            model_cam: self.processor.resize_image(
                model_cam, np.asarray(cameras[env_cam]["rgb"], dtype=np.uint8)
            )[None]
            for env_cam, model_cam in CAMERA_MAP.items()
        }

    def _to_vlam_obs(self, observation, eef_poses: np.ndarray, instruction: str):
        """Build the model input (history dim H=1) from the FK'd per-arm poses."""
        joints14 = self._joints14(observation)
        translation, quaternion = [], []
        for arm in range(self.NUM_ARMS):
            translation.append(eef_poses[arm, :3, 3])
            model_rotation = eef_poses[arm, :3, :3] @ EEF_TO_BASE_ROTATION
            quaternion.append(
                self._convert_rotation(
                    self._torch.from_numpy(model_rotation.reshape(1, 9)), self._quaternion
                )[0]
                .cpu()
                .numpy()
            )
        gripper = np.array(
            [joints14[self.GRIPPER_INDICES[arm]] for arm in range(self.NUM_ARMS)], dtype=np.float32
        )
        return self._VLAMRobotObservation(
            images=self._images(observation),
            instruction=instruction,
            ee_pose_translation=np.concatenate(translation).astype(np.float32)[None],  # [1, 6]
            ee_pose_rotation=np.concatenate(quaternion).astype(np.float32)[None],  # [1, 8]
            gripper=gripper[None],  # [1, 2]
            joints=self._per_arm_joints(observation).reshape(1, -1).astype(np.float32),  # [1, 12]
            timestamps=None,
            dataset_name=self.dataset_name,
        )

    # ---------------------------------------------------------------- inference

    def plan(self, observation, instruction: str):
        """One model query. Returns `(control, anchor_poses)`; anchor_poses is the snapshot all
        control-point deltas are measured from (see the module docstring)."""
        anchor_poses = self._eef_poses(observation)
        vlam_obs = self._to_vlam_obs(observation, anchor_poses, instruction)
        control = self.inference.step([vlam_obs])[0]
        return control, anchor_poses

    def qpos_for_step(self, control, anchor_poses: np.ndarray, timestep: int, observation) -> np.ndarray:
        """Target joints for control point `timestep`, anchored on `anchor_poses`.

        IK is warm-started from the *currently measured* joints so successive solutions stay on the
        same branch (the trajectory stays smooth); the target pose itself never depends on them.
        """
        current = self._per_arm_joints(observation)
        qpos = np.zeros(14, dtype=np.float32)
        for arm in range(self.NUM_ARMS):
            # Apply the delta in the model's pose space (A = R_fk @ M), then map the result back to
            # the FK frame the IK chain solves in. Translation is stored untransformed by the
            # converter, so it needs no such round trip.
            target = self._EndEffectorPoseAction(
                translation=anchor_poses[arm, :3, 3],
                rotation=anchor_poses[arm, :3, :3] @ EEF_TO_BASE_ROTATION,
            ).apply_delta(
                delta_rotation=control.eef_rotmat[arm, timestep].numpy(),
                delta_translation=control.eef_translation[arm, timestep].numpy(),
                delta_mode=self.delta_mode,
            )
            target_rotation = target.rotation.astype(np.float64) @ EEF_TO_BASE_ROTATION
            pose = self._pose_from_rotmat_translation(
                self._torch.from_numpy(target_rotation),
                self._torch.from_numpy(target.translation.astype(np.float64)),
            )
            qpos[self.ARM_SLICES[arm]] = (
                self._kinematics.ik(pose, self._torch.from_numpy(current[arm])).cpu().numpy()
            )
            # RoboTwin's Robot.set_gripper takes a normalized value in [0, 1], which is the same
            # convention the dataset's observation.gripper / control.gripper columns carry.
            qpos[self.GRIPPER_INDICES[arm]] = float(
                control.gripper_prob[arm, timestep, 0].clamp(0, 1)
            )
        return qpos

    def reset(self) -> None:
        """Nothing persists across episodes: each chunk is planned from a single observation."""


# ---------------------------------------------------------------------- RoboTwin hooks


def encode_obs(observation):
    """RoboTwin's observation is consumed directly (the policy owns FK and image resizing)."""
    return observation


def get_model(usr_args):
    ckpt_path = usr_args.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "ckpt_path is required: an absolute path to a checkpoint file inside a session tree laid "
            "out as <session>/checkpoints/<ckpt> alongside <session>/hf_export/<id>/src/"
        )
    return PiZeroVLAMPolicy(
        ckpt_path=ckpt_path,
        barrel_root=usr_args.get("barrel_root", "~/Documents/GitHub/barrel"),
        device=usr_args.get("device", "cuda"),
        dtype=usr_args.get("dtype", "bfloat16"),
        urdf_path=usr_args.get("urdf_path", ""),
        ee_link_name=usr_args.get("ee_link_name", ""),
        dataset_name=usr_args.get("dataset_name", "robotwin"),
        execute_steps=int(usr_args.get("execute_steps", 5)),
    )


def eval(TASK_ENV, model, observation):
    """One model query, then execute `execute_steps` of its control chunk.

    Called repeatedly by script/eval_policy.py until eval_success or take_action_cnt >= step_lim.
    """
    observation = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    control, anchor_poses = model.plan(observation, instruction)

    for timestep in range(model.execute_steps):
        qpos = model.qpos_for_step(control, anchor_poses, timestep, observation)
        TASK_ENV.take_action(qpos, action_type="qpos")
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            return
        # Refreshed only to warm-start the next step's IK; the anchor stays fixed for the chunk.
        observation = encode_obs(TASK_ENV.get_obs())


def reset_model(model):
    model.reset()
