from __future__ import annotations

import asyncio
import http
import socket

import numpy as np
import pytest
import torch
import websockets.asyncio.client as _client
import websockets.asyncio.server as _server
from websockets.exceptions import ConnectionClosedError

import deploy


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _observation() -> dict:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    return {
        "observation/state": np.arange(20, dtype=np.float32),
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/right_wrist_image": image,
        "prompt": "put shrimp in pot",
    }


class _FakeProcessor:
    def __init__(self) -> None:
        self.images = None
        self.prompt = None

    def __call__(self, *, images, language_instruction):
        self.images = images
        self.prompt = language_instruction
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "image_input": torch.zeros((1, 3, 3, 8, 12), dtype=torch.float32),
            "image_mask": torch.ones((1, 3), dtype=torch.bool),
        }


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = None
        self.steps = None

    def generate_actions(self, **kwargs):
        self.steps = kwargs.pop("steps")
        self.inputs = kwargs
        return torch.arange(600, dtype=torch.float32).reshape(1, 30, 20)


class _FakeWirePolicy:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.reset_calls = 0
        self.observations = []

    def reset(self) -> None:
        self.reset_calls += 1

    def infer(self, observation: dict) -> dict:
        self.observations.append(observation)
        if self.error is not None:
            raise self.error
        return {"actions": np.zeros((30, 20), dtype=np.float32)}


def test_arx_policy_builds_domain_19_inference():
    model = _FakeModel()
    processor = _FakeProcessor()
    policy = deploy.ARXXVLAPolicy(model, processor, steps=7)

    response = policy.infer(_observation())

    assert response["actions"].shape == (30, 20)
    assert response["actions"].dtype == np.float32
    assert processor.prompt == "put shrimp in pot"
    assert len(processor.images) == 3
    assert all(image.mode == "RGB" for image in processor.images)
    assert model.steps == 7
    assert model.inputs["proprio"].shape == (1, 20)
    assert model.inputs["domain_id"].tolist() == [19]


def test_arx_policy_rejects_bad_proprio_shape():
    policy = deploy.ARXXVLAPolicy(_FakeModel(), _FakeProcessor())
    observation = _observation()
    observation["observation/state"] = np.zeros(19, dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(20,\)"):
        policy.infer(observation)


def test_health_check_matches_pi05_contract():
    class Connection:
        @staticmethod
        def respond(status, body):
            return status, body

    class Request:
        path = "/healthz"

    assert deploy._health_check(Connection(), Request()) == (http.HTTPStatus.OK, "OK\n")
    Request.path = "/"
    assert deploy._health_check(Connection(), Request()) is None


def test_websocket_server_metadata_and_binary_response():
    async def run() -> None:
        policy = _FakeWirePolicy()
        port = _free_port()
        server = deploy.WebsocketPolicyServer(
            policy,
            host="127.0.0.1",
            port=port,
            metadata={"policy": "xvla_arx_a5"},
        )
        async with _server.serve(
            server._handler,
            "127.0.0.1",
            port,
            compression=None,
            max_size=None,
        ):
            async with _client.connect(
                f"ws://127.0.0.1:{port}",
                compression=None,
                max_size=None,
                proxy=None,
            ) as websocket:
                metadata = deploy.unpackb(await websocket.recv())
                assert metadata == {"policy": "xvla_arx_a5"}
                await websocket.send(deploy.packb(_observation()))
                response = deploy.unpackb(await websocket.recv())
                assert response["actions"].shape == (30, 20)
                assert response["server_timing"]["infer_ms"] >= 0.0
        assert policy.reset_calls == 1
        assert len(policy.observations) == 1

    asyncio.run(run())


def test_websocket_server_rejects_illegal_msgpack_like_pi05():
    async def run() -> None:
        port = _free_port()
        server = deploy.WebsocketPolicyServer(_FakeWirePolicy(), host="127.0.0.1", port=port)
        async with _server.serve(
            server._handler,
            "127.0.0.1",
            port,
            compression=None,
            max_size=None,
        ):
            async with _client.connect(
                f"ws://127.0.0.1:{port}",
                compression=None,
                max_size=None,
                proxy=None,
            ) as websocket:
                deploy.unpackb(await websocket.recv())
                await websocket.send(b"\xc1")
                error_frame = await websocket.recv()
                assert isinstance(error_frame, str)
                assert "Traceback" in error_frame
                with pytest.raises(ConnectionClosedError) as exc_info:
                    await websocket.recv()
                assert exc_info.value.rcvd.code == 1011

    asyncio.run(run())


def test_websocket_server_matches_pi05_error_sequence():
    async def run() -> None:
        policy = _FakeWirePolicy(error=ValueError("bad inference"))
        port = _free_port()
        server = deploy.WebsocketPolicyServer(policy, host="127.0.0.1", port=port)
        async with _server.serve(
            server._handler,
            "127.0.0.1",
            port,
            compression=None,
            max_size=None,
        ):
            async with _client.connect(
                f"ws://127.0.0.1:{port}",
                compression=None,
                max_size=None,
                proxy=None,
            ) as websocket:
                deploy.unpackb(await websocket.recv())
                await websocket.send(deploy.packb(_observation()))
                error_frame = await websocket.recv()
                assert isinstance(error_frame, str)
                assert "ValueError: bad inference" in error_frame
                with pytest.raises(ConnectionClosedError) as exc_info:
                    await websocket.recv()
                assert exc_info.value.rcvd.code == 1011

    asyncio.run(run())
