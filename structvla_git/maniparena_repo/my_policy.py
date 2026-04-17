"""EmuVLA adapter for ManipArena dual-arm end-effector control.

Usage:
    python serve.py \
        --checkpoint /path/to/emu_checkpoint \
        --control-mode end_pose \
        --action-horizon 10 \
        --port 8000

Optional environment variables:
    EMUVLA_VQ_HUB
    EMUVLA_VISION_HUB
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from maniparena.policy import ModelPolicy
from maniparena.utils import convert_model_output_to_action, convert_observation_to_model_input
from model_wrapper_emu import EmuVLAModel


DEFAULT_VQ_HUB = "/remote-home/linweihao/Emu3-Stage1"
DEFAULT_VISION_HUB = "/remote-home/linweihao/Emu3-VisionTokenizer"
DEFAULT_INSTRUCTION = "Move the block onto the designated colored square"


def _resolve_env_path(primary_key: str, legacy_key: str, default: str) -> str:
    return os.environ.get(primary_key) or os.environ.get(legacy_key) or default


class MyPolicy(ModelPolicy):
    def _ensure_image_save_state(self) -> None:
        if hasattr(self, "_input_image_step"):
            return
        self._input_image_step = 0
        self._input_image_dir = Path("my_policy_input_frames")
        self._input_image_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_uint8_image(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size > 0 and float(arr.max()) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _save_input_images(self, model_input: Dict[str, Any]) -> None:
        self._ensure_image_save_state()
        self._input_image_step += 1
        for key in ("front", "left", "right"):
            image = model_input.get(key)
            if image is None:
                continue
            arr = self._to_uint8_image(image)
            out_path = self._input_image_dir / f"step_{self._input_image_step:06d}_{key}.jpg"
            Image.fromarray(arr).save(out_path, format="JPEG", quality=95)

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        if self.control_mode != "end_pose":
            raise ValueError(
                "EmuVLA ManipArena adapter currently supports only --control-mode end_pose."
            )

        vq_hub = _resolve_env_path("EMUVLA_VQ_HUB", "STRUCTVLA_VQ_HUB", DEFAULT_VQ_HUB)
        vision_hub = _resolve_env_path("EMUVLA_VISION_HUB", "STRUCTVLA_VISION_HUB", DEFAULT_VISION_HUB)

        model = EmuVLAModel(
            emu_hub=checkpoint_path,
            vq_hub=vq_hub,
            vision_hub=vision_hub,
            device=device,
        )
        model.predict_action_frames = self.action_horizon
        return model

    def run_inference(self, model_input: Dict[str, Any]) -> list[list[float]]:
        for key in ("front", "left", "right"):
            if model_input.get(key) is None:
                raise ValueError(f"Missing required camera view '{key}' for EmuVLA inference.")

        #self._save_input_images(model_input)

        instruction = str(model_input.get("instruction", DEFAULT_INSTRUCTION) or DEFAULT_INSTRUCTION)
        print(instruction)
        actions = self.model.step(
            image={
                "front": model_input["front"],
                "left": model_input["left"],
                "right": model_input["right"],
            },
            goal=instruction,
        )
        actions = np.asarray(actions, dtype=np.float32)
        return actions[: self.action_horizon].tolist()

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return convert_observation_to_model_input(
            obs,
            self.control_mode,
            decode_images=True,
        )

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        actions = np.asarray(model_output, dtype=np.float32)
        return convert_model_output_to_action(
            actions,
            self.control_mode,
            self.action_horizon,
        )
