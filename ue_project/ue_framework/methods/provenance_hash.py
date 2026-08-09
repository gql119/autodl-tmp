from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def portable_recipe_sha256(
    *,
    schema: str,
    parameters: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> str:
    """Hash frozen inputs and parameters without hashing platform float bytes."""
    if not str(schema).strip():
        raise ValueError("Recipe hash schema must be non-empty.")
    if not source_provenance:
        raise ValueError("Recipe hash requires non-empty source provenance.")
    payload = {
        "schema": str(schema),
        "parameters": dict(parameters),
        "source_provenance": dict(source_provenance),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
