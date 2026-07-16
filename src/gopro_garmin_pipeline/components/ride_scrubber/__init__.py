"""Ride scrubber component — slider with live frame preview."""

from pathlib import Path

import streamlit.components.v1 as components

_component_func = components.declare_component(
    "ride_scrubber",
    path=str(Path(__file__).parent),
)


def ride_scrubber(
    frame_urls: list[str],
    time_labels: list[str],
    ride_secs: list[int],
    clip_names: list[str],
    initial_idx: int = 0,
    key: str | None = None,
) -> int | None:
    """Render a ride scrubber with live frame preview during drag.

    Returns the ride_secs value of the selected frame on release,
    or None if no interaction yet.
    """
    result = _component_func(
        frameUrls=frame_urls,
        timeLabels=time_labels,
        rideSecs=ride_secs,
        clipNames=clip_names,
        initialIdx=initial_idx,
        key=key,
        default=None,
    )
    return result
