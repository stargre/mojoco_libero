"""Small, dependency-light episode aggregation helpers."""


def summarize_episodes(episodes):
    """Aggregate episode dictionaries containing suite and success fields."""
    total = len(episodes)
    successes = sum(
        1
        for episode in episodes
        if bool(episode.get("episode_success", episode.get("success", False)))
    )

    by_suite = {}
    for episode in episodes:
        suite = str(episode.get("suite", "unknown"))
        bucket = by_suite.setdefault(suite, {"episodes": 0, "successes": 0})
        bucket["episodes"] += 1
        if bool(episode.get("episode_success", episode.get("success", False))):
            bucket["successes"] += 1

    for bucket in by_suite.values():
        bucket["success_rate"] = (
            bucket["successes"] / bucket["episodes"]
            if bucket["episodes"]
            else 0.0
        )

    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "by_suite": by_suite,
    }
