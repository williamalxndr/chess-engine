def resolve_latest(prefix: str, registry: dict[str, type]) -> str:
    """
    Auto-detect the highest versioned class matching prefix from a registry.

    Args:
        prefix:   Class name prefix to match, e.g. "ChessEncoder", "ChessNetwork"
        registry: Dict mapping class name → class, populated via __init_subclass__

    Returns:
        Name of the class with the highest version suffix.

    Example:
        resolve_latest("ChessEncoder", ChessEncoder._registry)  # → "ChessEncoderV2"
        resolve_latest("ChessNetwork", ChessNetwork._registry)  # → "ChessNetworkV3"
    """
    def _parse_version(suffix: str) -> int:
        try:
            return int(suffix.lstrip("Vv"))
        except ValueError:
            return -1

    candidates = {
        name: _parse_version(name[len(prefix):])
        for name in registry
        if name.startswith(prefix)
    }

    if not candidates:
        raise ValueError(f"No classes found for prefix '{prefix}' in registry")

    return max(candidates, key=candidates.get)

