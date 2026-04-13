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
        use_ensemble: bool = True,
        ensemble_m: float = 0.01,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._use_ensemble = use_ensemble
        self._ensemble_m = ensemble_m
        self._ensemble_data = {}
        self._current_step = 0
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

        # Reset ensemble state for each new connection
        self._ensemble_data = {}
        self._current_step = 0
        
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action_chunk = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                if self._use_ensemble:
                    # 1. Add the new action chunk to the buffer with weights
                    for i in range(len(action_chunk["action"])):
                        target_step = self._current_step + i
                        weight = np.exp(-self._ensemble_m * i)
                        
                        current_val, current_weight = self._ensemble_data.get(target_step, (0, 0))
                        self._ensemble_data[target_step] = (current_val + action_chunk["action"][i] * weight, current_weight + weight)

                    # 2. Get the smoothed action for the current step
                    final_action_sum, final_weight_sum = self._ensemble_data[self._current_step]
                    final_action = final_action_sum / final_weight_sum
                    
                    # Replace the chunk with the single smoothed action
                    action = {"action": final_action}

                    # 3. Clean up old buffer data
                    del self._ensemble_data[self._current_step]
                    
                    self._current_step += 1
                else:
                    # Original behavior: return the first action of the chunk
                    action = {"action": action_chunk["action"][0]}


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
