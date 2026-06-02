#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

"""Serve the ARX-A5 X-VLA checkpoint on the existing robot rollout wire."""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import os.path as osp
import sys
import time
import traceback
from typing import Any

import msgpack
import numpy as np
from PIL import Image
import torch
import websockets.asyncio.server as _server
from websockets.exceptions import ConnectionClosed
import websockets.frames


ARX_DOMAIN_ID = 19
ARX_ACTION_DIM = 20
ARX_CAMERA_KEYS = (
    "observation/image",
    "observation/wrist_image",
    "observation/right_wrist_image",
)

logger = logging.getLogger("xvla.deploy")


# Keep this codec wire-compatible with openpi_client.msgpack_numpy without
# importing the PI05 source tree into the X-VLA environment.
def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class ARXXVLAPolicy:
    """Translate the robot rollout observation into one X-VLA inference."""

    def __init__(self, model, processor, *, steps: int = 10) -> None:
        if processor is None:
            raise ValueError("XVLAProcessor is required for ARX-A5 deployment")
        if int(steps) <= 0:
            raise ValueError("steps must be positive")
        self.model = model
        self.processor = processor
        self.steps = int(steps)

    def reset(self) -> None:
        """X-VLA inference is stateless across WebSocket connections."""

    def infer(self, observation: dict) -> dict[str, np.ndarray]:
        proprio = np.asarray(observation["observation/state"], dtype=np.float32)
        if proprio.shape != (ARX_ACTION_DIM,):
            raise ValueError(
                f"observation/state must have shape ({ARX_ACTION_DIM},), "
                f"got {proprio.shape}"
            )

        images = [self._as_rgb_image(observation[key], key=key) for key in ARX_CAMERA_KEYS]
        inputs = self.processor(
            images=images,
            language_instruction=observation["prompt"],
        )
        required = {"input_ids", "image_input", "image_mask"}
        if not required.issubset(inputs):
            raise ValueError(
                f"processor returned incomplete inputs: missing {sorted(required - set(inputs))}"
            )

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        def to_model(value) -> torch.Tensor:
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.is_floating_point():
                return tensor.to(device=device, dtype=dtype)
            return tensor.to(device=device)

        model_inputs = {key: to_model(value) for key, value in inputs.items()}
        model_inputs.update(
            {
                "proprio": to_model(torch.from_numpy(proprio).unsqueeze(0)),
                "domain_id": torch.tensor([ARX_DOMAIN_ID], dtype=torch.long, device=device),
            }
        )
        with torch.inference_mode():
            actions = self.model.generate_actions(
                **model_inputs,
                steps=self.steps,
            )
        actions = actions.squeeze(0).float().cpu().numpy()
        if actions.ndim != 2 or actions.shape[1] != ARX_ACTION_DIM:
            raise ValueError(
                f"X-VLA actions must have shape (T, {ARX_ACTION_DIM}), "
                f"got {actions.shape}"
            )
        return {"actions": actions.astype(np.float32, copy=False)}

    @staticmethod
    def _as_rgb_image(value, *, key: str) -> Image.Image:
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{key} must have HxWx3 shape, got {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(image), mode="RGB")


class WebsocketPolicyServer:
    """OpenPI-compatible WebSocket server used by the existing robot client."""

    def __init__(
        self,
        policy: ARXXVLAPolicy,
        *,
        host: str = "0.0.0.0",
        port: int = 8010,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = str(host)
        self._port = int(port)
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        self._policy.reset()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                observation = unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(observation)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _metadata(model) -> dict[str, Any]:
    return {
        "policy": "xvla_arx_a5",
        "domain_id": ARX_DOMAIN_ID,
        "action_mode": "arx_ee6d",
        "action_dim": ARX_ACTION_DIM,
        "chunk_size": int(model.config.num_actions),
    }


def _load_model_and_processor(args):
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    processor_path = args.processor_path or args.model_path
    logger.info("Loading XVLAProcessor from %s", processor_path)
    processor = XVLAProcessor.from_pretrained(processor_path)

    logger.info("Loading XVLA checkpoint from %s", args.model_path)
    model = XVLA.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device).to(torch.float32)
    if args.LoRA_path is not None:
        from peft import PeftModel

        logger.info("Applying LoRA weights from %s", args.LoRA_path)
        model = PeftModel.from_pretrained(
            model,
            args.LoRA_path,
            torch_dtype=torch.float32,
        ).to(args.device)
    model.eval()
    return model, processor


def _resolve_host(args) -> tuple[str, str, str | None]:
    node_list = os.environ.get("SLURM_NODELIST")
    job_id = os.environ.get("SLURM_JOB_ID", "none")
    if node_list and not args.disable_slurm:
        host = ".".join(node_list.split("-")[1:]) if "-" in node_list else node_list
        return host, job_id, node_list
    return args.host, job_id, node_list


def _write_info(output_dir: str, *, host: str, port: int, job_id: str, node_list: str | None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    info_path = osp.join(output_dir, "info.json")
    if osp.exists(info_path):
        raise FileExistsError(
            f"{info_path} already exists; remove it or use a different --output_dir"
        )
    with open(info_path, "w") as f:
        json.dump(
            {
                "host": host,
                "port": int(port),
                "job_id": job_id,
                "node_list": node_list or "none",
            },
            f,
            indent=4,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch ARX-A5 X-VLA WebSocket server")
    parser.add_argument("--model_path", required=True, help="XVLA checkpoint directory")
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--LoRA_path", default=None)
    parser.add_argument("--output_dir", default="./logs")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--port", default=8010, type=int)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--steps", default=10, type=int, help="Diffusion inference steps")
    parser.add_argument("--disable_slurm", action="store_true")
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    host, job_id, node_list = _resolve_host(args)
    model, processor = _load_model_and_processor(args)
    if str(model.config.action_mode).lower() != "arx_ee6d":
        raise ValueError(
            f"ARX-A5 checkpoint must use action_mode=arx_ee6d, "
            f"got {model.config.action_mode}"
        )
    if int(model.config.num_actions) != 30:
        raise ValueError(
            f"ARX-A5 checkpoint must predict 30 actions, got {model.config.num_actions}"
        )
    _write_info(
        args.output_dir,
        host=host,
        port=args.port,
        job_id=job_id,
        node_list=node_list,
    )
    policy = ARXXVLAPolicy(model, processor, steps=args.steps)
    server = WebsocketPolicyServer(
        policy,
        host=host,
        port=args.port,
        metadata=_metadata(model),
    )
    logger.info("Serving ARX-A5 X-VLA at ws://%s:%d", host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
