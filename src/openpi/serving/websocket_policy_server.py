import asyncio
import http
import logging
import time
import traceback
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
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
        execute_k: int = 15,
        overlap_new_weight: float = 0.75,
        use_vp_noise: bool = True,
        noise_beta: float = 0.5,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._execute_k = execute_k
        self._overlap_new_weight = overlap_new_weight
        self._use_vp_noise = use_vp_noise
        self._noise_beta = noise_beta
        self._noise_sigma = float(np.sqrt(max(0.0, 1.0 - noise_beta**2)))
        self._action_buffer: list[np.ndarray] = []
        self._overlap_buffer: np.ndarray | None = None
        self._prev_noise: np.ndarray | None = None
        self._last_episode_index: int | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def _reset_temporal_state(self) -> None:
        self._action_buffer = []
        self._overlap_buffer = None
        self._prev_noise = None

    def _should_reset_from_obs(self, obs: dict) -> bool:
        # Common reset flags used by env wrappers.
        for key in ("reset", "is_first", "new_episode", "episode_start"):
            if bool(obs.get(key, False)):
                return True

        # If frame index is explicitly provided, treat frame 0 as a fresh episode.
        frame_index = obs.get("frame_index")
        if frame_index is not None:
            try:
                if int(frame_index) == 0:
                    return True
            except (TypeError, ValueError):
                pass

        # Reset when episode id changes.
        episode_index = obs.get("episode_index")
        if episode_index is not None:
            try:
                epi = int(episode_index)
                if self._last_episode_index is None:
                    self._last_episode_index = epi
                elif epi != self._last_episode_index:
                    self._last_episode_index = epi
                    return True
            except (TypeError, ValueError):
                pass

        return False

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

        # Reset per-connection temporal state.
        self._reset_temporal_state()
        self._last_episode_index = None
        
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if isinstance(obs, dict) and self._should_reset_from_obs(obs):
                    self._reset_temporal_state()
                    # Keep routing / policy state clean if policy exposes a reset hook.
                    if hasattr(self._policy, "reset") and callable(getattr(self._policy, "reset")):
                        self._policy.reset()

                infer_time = 0.0
                if not self._action_buffer:
                    infer_start = time.monotonic()
                    if self._use_vp_noise and self._prev_noise is not None:
                        try:
                            infer_result = self._policy.infer(obs, noise=self._prev_noise)
                        except TypeError:
                            infer_result = self._policy.infer(obs)
                    else:
                        infer_result = self._policy.infer(obs)
                    infer_time = time.monotonic() - infer_start

                    action_chunk = infer_result.get("actions", infer_result.get("action"))
                    if action_chunk is None:
                        raise ValueError("Policy output must include 'actions' or 'action'.")

                    chunk_arr = np.asarray(action_chunk)
                    if chunk_arr.ndim == 1:
                        chunk_arr = chunk_arr[None, :]

                    if self._use_vp_noise:
                        # Generate warm-start noise in float32; model side will cast to runtime dtype.
                        eps = np.random.randn(*chunk_arr.shape).astype(np.float32)
                        if self._prev_noise is None or self._prev_noise.shape != chunk_arr.shape:
                            self._prev_noise = eps
                        else:
                            self._prev_noise = (
                                self._noise_beta * self._prev_noise
                                + self._noise_sigma * eps
                            )

                    if self._overlap_buffer is not None and len(self._overlap_buffer) > 0:
                        overlap_len = min(len(self._overlap_buffer), len(chunk_arr))
                        old_weight = 1.0 - self._overlap_new_weight
                        chunk_arr[:overlap_len] = (
                            old_weight * self._overlap_buffer[:overlap_len]
                            + self._overlap_new_weight * chunk_arr[:overlap_len]
                        )

                    k = max(1, min(self._execute_k, len(chunk_arr)))
                    self._action_buffer = [chunk_arr[i] for i in range(k)]
                    self._overlap_buffer = chunk_arr[k:] if k < len(chunk_arr) else None

                # PiPolicy / genie_sim expect `actions` to be a sequence of per-step vectors
                # (deque iterates the outer sequence). A single 1D ndarray would iterate as
                # per-dimension scalars — wrap as a one-element list.
                step = np.asarray(self._action_buffer.pop(0))
                action = {"actions": [step]}


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


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
