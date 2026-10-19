"""Deterministic group-level dev/test split (no task appears in both)."""
import hashlib


def group_hash(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def assign(groups, dev_fraction: float, seed: int) -> dict:
    """Map each group to "dev" or "test"; ordering is by seeded hash."""
    ordered = sorted(set(groups), key=lambda g: group_hash(g, seed))
    n_dev = round(len(ordered) * dev_fraction)
    return {g: ("dev" if i < n_dev else "test") for i, g in enumerate(ordered)}
