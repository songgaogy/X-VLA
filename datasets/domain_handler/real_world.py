# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, Iterable
import numpy as np
import h5py
import random
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d
from ..utils import quat_to_rotate6d, read_parquet, read_video_to_frames
from .base import BaseHDF5Handler, DomainHandler


class AIRAgilexHandler(BaseHDF5Handler):
    """
    AIR-AGILEX (non-HQ).
    HDF5:
      /observations/eef_quaternion [T, 16] =
        L_xyz(3) L_quat(4) L_grip_raw(1) R_xyz(3) R_quat(4) R_grip_raw(1)
    Output: left/right [T,10] = xyz(3)+rot6d(6)+grip(1), grip thresholded.
    """
    dataset_name = "AIR-AGILEX"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 2.0
        eef = f["observations/eef_quaternion"][()]  # [T,16]
        l_xyz, l_quat, l_grip = eef[:, :3], eef[:, 3:7], (eef[:, 7:8] * 50 < 1.0)
        r_xyz, r_quat, r_grip = eef[:, 8:11], eef[:, 11:15], (eef[:, 15:16] * 50 < 1.0)
        left  = np.concatenate([l_xyz, quat_to_rotate6d(l_quat), l_grip], axis=-1)
        right = np.concatenate([r_xyz, quat_to_rotate6d(r_quat), r_grip], axis=-1)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        # stride 2 for denser lookahead; leave 30 frames margin
        return range(0, max(0, T_left - 30), 2)


class AIRAgilexHQHandler(BaseHDF5Handler):
    """
    AIR-AGILEX-HQ.
    HDF5:
      /observations/eef_6d        [T,20]  -> L(10)+R(10)
      /observations/eef_left_time [T]
      /observations/eef_right_time[T]
    Grip thresholded from last channel (scaled by 50).
    """
    dataset_name = "AIR-AGILEX-HQ"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 2.0
        eef = f["observations"]["eef_6d"][()]  # [T,20]
        left, right = eef[:, :10], eef[:, 10:]
        left[:,  -1] = (left[:,  -1] * 50 < 1.0)
        right[:, -1] = (right[:, -1] * 50 < 1.0)
        lt = f["/observations/eef_left_time"][()]
        rt = f["/observations/eef_right_time"][()]
        f.close()
        return left, right, lt, rt, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        index =  list(range(0, max(0, T_left - 60)))
        if training: random.shuffle(index)
        return index



class AIRARXHandler(DomainHandler):
    """
    ARX-A5 dual-arm LeRobot v3.0 dataset.

    Parquet:
      observation.state [T,20] -> current proprio
      action            [T,20] -> L(10)+R(10) continuous EE6D targets

    Output trajectories contain the current proprio followed by one second of
    future actions. The shared dataset layer splits this into proprio and action.
    """
    dataset_name = "ARX-A5"

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        tasks_path = fileio.join_path(meta["root_path"], meta["tasks_path"])
        tasks = read_parquet(tasks_path)
        self.task_by_index = dict(zip(tasks["task_index"], tasks["task"]))

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        if action_mode.lower() != "arx_ee6d":
            raise ValueError("ARX-A5 requires action_mode='arx_ee6d' for continuous gripper targets")

        item = self.meta["datalist"][traj_idx]
        episode_index = int(item["episode_index"])
        chunk_index = episode_index // int(self.meta["chunks_size"])
        format_args = {
            "chunk_index": chunk_index,
            "file_index": episode_index,
        }

        data_path = fileio.join_path(
            self.meta["root_path"],
            self.meta["data_path"].format(**format_args),
        )
        data = read_parquet(data_path)
        states = np.asarray(data[self.meta["state_key"]], dtype=np.float64)
        actions = np.asarray(data[self.meta["action_key"]], dtype=np.float64)
        timestamps = np.asarray(data["timestamp"], dtype=np.float64)
        task_indices = np.asarray(data["task_index"], dtype=np.int64)
        if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 20:
            raise ValueError(
                f"Expected matching [T,20] state/action arrays, got {states.shape} and {actions.shape}"
            )

        images = [
            read_video_to_frames(
                fileio.join_path(
                    self.meta["root_path"],
                    self.meta["video_path"].format(video_key=video_key, **format_args),
                )
            )
            for video_key in self.meta["camera_views"][:self.num_views]
        ]
        if any(len(frames) != len(states) for frames in images):
            raise ValueError(f"Video/parquet frame count mismatch in episode {episode_index}")

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:len(images)] = True
        freq = float(self.meta["fps"])
        qdur = 1.0
        idxs = list(range(0, max(0, len(actions) - int(round(freq * qdur)))))
        if training:
            random.shuffle(idxs)

        action_interp = interp1d(
            timestamps,
            actions,
            axis=0,
            bounds_error=False,
            fill_value=(actions[0], actions[-1]),
        )
        for idx in idxs:
            query = np.linspace(
                timestamps[idx] + 1.0 / freq,
                timestamps[idx] + qdur,
                num_actions,
                dtype=np.float64,
            )
            trajectory = np.concatenate([states[idx:idx + 1], action_interp(query)], axis=0)
            imgs = [image_aug(Image.fromarray(frames[idx])) for frames in images]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            instruction = self.task_by_index[int(task_indices[idx])]
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])
            yield {
                "language_instruction": instruction,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": torch.from_numpy(trajectory).float(),
            }

class AIRBotHandler(BaseHDF5Handler):
    """
    AIRBOT.
    HDF5:
      /eef_6d [T,10] -> xyz(3)+rot6d(6)+grip_raw(1)
    Single arm (left), right is zeros. Grip <0.5 => closed.
    """
    dataset_name = "AIRBOT"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 10.0, 3.0
        eef = f["eef_6d"][()]  # [T,10]
        left = np.concatenate([eef[:, :9], (eef[:, 9:] < 0.5)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


class WidowxAirHandler(BaseHDF5Handler):
    """
    widowx-air.
    HDF5:
      /abs_action_6d [T,10] -> xyz(3)+rot6d(6)+grip_raw(1)
    Single arm; grip <0.5 => closed.
    """
    dataset_name = "widowx-air"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 5.0, 5.0
        a = f["abs_action_6d"][()]  # [T,10]
        left = np.concatenate([a[:, :9], (a[:, 9:] < 0.5)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 15))
