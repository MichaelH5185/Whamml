"""
dedup.py
--------
Deduplication of detection pairs by building identity.

Because a building can appear in many frames (potentially hundreds), we cap
the number of retained pairs per building at *max_per_building*.

Strategy
--------
Detection pairs are keyed by (building_lon, building_lat) — the exact
coordinates looked up from the building CSV.  All pairs for the same key
are grouped together, and up to *max_per_building* are kept, preferring
those with the highest IoU score (best geometric match quality).
"""

from __future__ import annotations
import random
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .matcher import DetectionPair


def deduplicate(
    pairs: list["DetectionPair"],
    max_per_building: int = 3,
    seed: int = 42,
) -> list["DetectionPair"]:
    """
    Group *pairs* by building (lon, lat) key and keep the top-N by IoU score.

    Parameters
    ----------
    pairs            : All candidate DetectionPair objects (with damage_class set).
    max_per_building : Maximum pairs to retain per unique building.
    seed             : Random seed for tie-breaking.

    Returns
    -------
    Deduplicated list of DetectionPairs.
    """
    random.seed(seed)

    # Group by building coordinate key
    groups: dict[tuple, list["DetectionPair"]] = defaultdict(list)
    ungrouped: list["DetectionPair"] = []

    for pair in pairs:
        if pair.building_lon is not None and pair.building_lat is not None:
            key = (round(pair.building_lon, 7), round(pair.building_lat, 7))
            groups[key].append(pair)
        else:
            ungrouped.append(pair)

    kept: list["DetectionPair"] = []

    for key, group in groups.items():
        # Sort by IoU descending (highest quality first)
        group_sorted = sorted(group, key=lambda p: p.iou, reverse=True)
        kept.extend(group_sorted[:max_per_building])

    # Ungrouped pairs (no building assignment) should already be filtered
    # upstream, but add them as a safety net
    kept.extend(ungrouped)

    return kept
