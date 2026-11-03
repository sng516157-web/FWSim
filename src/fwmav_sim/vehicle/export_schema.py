"""Export the Pydantic JSON Schema for VehicleDefinition to docs/schema/.

Run from anywhere after ``pip install -e .[dev]``:

    python -m fwmav_sim.vehicle.export_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from fwmav_sim.vehicle.schema import VehicleDefinition

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT = _REPO_ROOT / "docs" / "schema" / "vehicle_definition.schema.json"


def main() -> Path:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(VehicleDefinition.model_json_schema(), indent=2) + "\n")
    return _OUT


if __name__ == "__main__":
    print(main())
