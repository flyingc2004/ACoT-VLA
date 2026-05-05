"""将 websocket 传入的 obs 路由到对应 checkpoint 上的 Policy。"""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from openpi_client import base_policy as _base_policy

from openpi.policies.checkpoint_switcher import CheckpointRoutingSwitcher


class RoutingPolicy(_base_policy.BasePolicy):
    """实现 BasePolicy：每次 infer 根据 obs 选择底层 Policy。"""

    def __init__(
        self,
        switcher: CheckpointRoutingSwitcher,
        *,
        preload_default: bool = False,
    ) -> None:
        self._switcher = switcher
        self._metadata = switcher.default_server_metadata()
        if preload_default:
            # 触发默认权重加载，便于启动即占满显存、尽早暴露路径错误
            self._switcher.get_policy_for_obs({"prompt": "", "task": ""})

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        policy = self._switcher.get_policy_for_obs(obs)
        return policy.infer(obs, **kwargs)

    @override
    def reset(self) -> None:
        self._switcher.reset_current_policy()
