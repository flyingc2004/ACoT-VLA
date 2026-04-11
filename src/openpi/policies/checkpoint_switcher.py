"""按观测（task / prompt）在多个 checkpoint 间切换，同一时间仅保留一份模型在显存中。"""

from __future__ import annotations

import gc
import json
import logging
import pathlib
from typing import Any

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)


def _scalar_to_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


class CheckpointRoutingSwitcher:
    """读取路由 JSON，按需加载 checkpoint；切换时卸载当前 policy 并清理 JAX 缓存。"""

    def __init__(
        self,
        routing_json_path: str | pathlib.Path,
        *,
        default_prompt: str | None = None,
        strict_load: bool = False,
    ) -> None:
        self._routing_path = pathlib.Path(routing_json_path).expanduser().resolve()
        self._default_prompt = default_prompt
        self._strict_load_override = strict_load

        self._default_key: str
        self._checkpoints: dict[str, dict[str, str]]
        self._route_task: dict[str, str]
        self._route_prompt_exact: dict[str, str]
        self._route_prompt_contains: list[tuple[str, str]]
        self._strict_from_file: bool

        self._load_and_validate()

        self._current_policy: _policy.Policy | None = None
        self._current_key: str | None = None

    def _load_and_validate(self) -> None:
        try:
            with self._routing_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            logger.error("Checkpoint routing file not found: %s", self._routing_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in routing file: %s", e)
            raise

        if "checkpoints" not in raw or not isinstance(raw["checkpoints"], dict):
            raise ValueError("Routing file must contain a non-empty 'checkpoints' object")

        default_key = raw.get("default_checkpoint")
        if not default_key or not isinstance(default_key, str):
            raise ValueError("Routing file must set string 'default_checkpoint'")

        checkpoints: dict[str, dict[str, str]] = {}
        for name, info in raw["checkpoints"].items():
            if not isinstance(info, dict):
                raise ValueError(f"Checkpoint '{name}' must be an object")
            cfg = info.get("config")
            path = info.get("path")
            if not cfg or not path:
                raise ValueError(f"Checkpoint '{name}' requires 'config' and 'path'")
            checkpoints[name] = {"config": str(cfg), "path": str(path)}

        if default_key not in checkpoints:
            raise ValueError(f"default_checkpoint '{default_key}' is not defined in 'checkpoints'")

        self._default_key = default_key
        self._checkpoints = checkpoints
        self._route_task = _normalize_str_dict(raw.get("route_by_task_name", {}))
        self._route_prompt_exact = _normalize_str_dict(raw.get("route_by_prompt_exact", {}))

        contains = raw.get("route_by_prompt_contains", {})
        if not isinstance(contains, dict):
            raise ValueError("'route_by_prompt_contains' must be an object")
        self._route_prompt_contains = [(str(k), str(v)) for k, v in contains.items()]
        for _, ckpt_key in self._route_prompt_contains:
            if ckpt_key not in checkpoints:
                raise ValueError(f"route_by_prompt_contains references unknown checkpoint '{ckpt_key}'")

        for m in (self._route_task, self._route_prompt_exact):
            for _, ckpt_key in m.items():
                if ckpt_key not in checkpoints:
                    raise ValueError(f"route map references unknown checkpoint '{ckpt_key}'")

        self._strict_from_file = bool(raw.get("strict_load", False))

        logger.info(
            "Checkpoint routing loaded from %s (%d checkpoints, default=%s)",
            self._routing_path,
            len(checkpoints),
            default_key,
        )

    @property
    def strict_load(self) -> bool:
        return self._strict_load_override or self._strict_from_file

    def default_server_metadata(self) -> dict[str, Any]:
        cfg_name = self._checkpoints[self._default_key]["config"]
        cfg = _config.get_config(cfg_name)
        return dict(cfg.policy_metadata or {})

    def resolve_checkpoint_key(self, obs: dict) -> str:
        task = ""
        for key in ("task", "task_name"):
            if key in obs:
                task = _scalar_to_str(obs[key]).strip()
                break
        if task and task in self._route_task:
            return self._route_task[task]

        prompt = _scalar_to_str(obs.get("prompt", "")).strip()
        if prompt and prompt in self._route_prompt_exact:
            return self._route_prompt_exact[prompt]

        prompt_lower = prompt.lower()
        for needle, ckpt_key in self._route_prompt_contains:
            if needle.lower() in prompt_lower:
                return ckpt_key

        if task or prompt:
            logger.warning(
                "No routing rule matched (task=%r, prompt prefix=%r); using default %r",
                task,
                prompt[:80] if prompt else "",
                self._default_key,
            )
        return self._default_key

    def _unload_current(self) -> None:
        if self._current_policy is None:
            return
        logger.info("Unloading checkpoint %r", self._current_key)
        del self._current_policy
        self._current_policy = None
        self._current_key = None
        gc.collect()
        try:
            import jax

            jax.clear_caches()
            logger.info("Cleared JAX caches after unload")
        except Exception as e:
            logger.warning("Could not clear JAX caches: %s", e)

    def _load_checkpoint(self, key: str) -> _policy.Policy:
        entry = self._checkpoints[key]
        train_cfg = _config.get_config(entry["config"])
        path = entry["path"]
        logger.info("Loading checkpoint %r from %s (config=%s)", key, path, entry["config"])
        return _policy_config.create_trained_policy(
            train_cfg,
            path,
            default_prompt=self._default_prompt,
        )

    def get_policy_for_obs(self, obs: dict) -> _policy.Policy:
        target_key = self.resolve_checkpoint_key(obs)

        if self._current_key == target_key and self._current_policy is not None:
            return self._current_policy

        self._unload_current()

        try:
            self._current_policy = self._load_checkpoint(target_key)
            self._current_key = target_key
            logger.info("Active checkpoint: %r", target_key)
            return self._current_policy
        except Exception as e:
            logger.error("Failed to load checkpoint %r: %s", target_key, e)
            if self.strict_load or target_key == self._default_key:
                raise
            logger.warning("Falling back to default checkpoint %r", self._default_key)
            self._current_policy = self._load_checkpoint(self._default_key)
            self._current_key = self._default_key
            return self._current_policy

    def reset_current_policy(self) -> None:
        if self._current_policy is not None:
            self._current_policy.reset()


def _normalize_str_dict(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("route map must be an object")
    return {str(k): str(v) for k, v in raw.items()}
