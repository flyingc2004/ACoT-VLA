import asyncio
import http
import logging
import os
import pathlib
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from PIL import Image
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._tophead_record_dir = os.getenv("TOPHEAD_RECORD_DIR")
        self._tophead_record_every = max(1, int(os.getenv("TOPHEAD_RECORD_EVERY", "1")))
        self._tophead_record_sequence = os.getenv("TOPHEAD_RECORD_SEQUENCE", "0").lower() in ("1", "true", "yes", "on")
        self._tophead_record_step = 0
        if self._tophead_record_dir:
            pathlib.Path(self._tophead_record_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                "Updating top_head image at %s/tophead.png every %d frame(s)",
                self._tophead_record_dir,
                self._tophead_record_every,
            )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                self._maybe_record_tophead(obs)

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _maybe_record_tophead(self, obs: dict) -> None:
        if not self._tophead_record_dir:
            return

        step = self._tophead_record_step
        self._tophead_record_step += 1
        if step % self._tophead_record_every != 0:
            return

        image = _extract_tophead_image(obs)
        if image is None:
            if step == 0:
                logger.warning("TOPHEAD_RECORD_DIR is set, but no top_head image was found in the first observation.")
            return

        image = _to_uint8_hwc(image)
        if image is None:
            if step == 0:
                logger.warning("TOPHEAD_RECORD_DIR is set, but top_head image has an unsupported shape/type.")
            return

        output_dir = pathlib.Path(self._tophead_record_dir)
        Image.fromarray(image).save(output_dir / "tophead.png")
        if self._tophead_record_sequence:
            Image.fromarray(image).save(output_dir / f"top_head_{step:06d}.jpg", quality=90)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _extract_tophead_image(obs: dict):
    images = obs.get("images")
    if isinstance(images, dict) and "top_head" in images:
        return images["top_head"]

    observation = obs.get("observation")
    if isinstance(observation, dict):
        observation_images = observation.get("images")
        if isinstance(observation_images, dict) and "top_head" in observation_images:
            return observation_images["top_head"]

    return None


def _to_uint8_hwc(image) -> np.ndarray | None:
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        return None

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        return None

    if np.issubdtype(array.dtype, np.floating):
        max_value = np.nanmax(array) if array.size else 0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)

    return np.clip(array, 0, 255).astype(np.uint8)
