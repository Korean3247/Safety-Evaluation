from framing_sensitivity.constants import FRAME_TEMPLATES


def build_framed_prompt(original_prompt: str, frame: str) -> str:
    if frame not in FRAME_TEMPLATES:
        raise ValueError(f"Unsupported frame: {frame}")
    return FRAME_TEMPLATES[frame].format(original_prompt=original_prompt)
