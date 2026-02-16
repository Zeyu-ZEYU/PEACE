from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def interp_piecewise(points: List[Tuple[float, float]], x: float) -> float:
    """Piecewise-linear interpolation with endpoint clamping."""
    if not points:
        raise ValueError("empty curve")
    pts = sorted(points, key=lambda p: p[0])

    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])

    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return float(y1)
            t = (x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

    # Fallback (should not happen).
    return float(pts[-1][1])


@dataclass(frozen=True)
class ProfileDB:
    """In-memory database of profiled curves.

    JSON schema:
    {
      "methods": {
        "sarathi": {
          "glm-4": {
            "needlebench": {
              "ttft_s": [[x,y], ...],
              "tbt_s": [[x,y], ...],
              ...
            }
          }
        }
      }
    }
    """

    methods: Dict[str, Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]]

    def curve_points(self, method: str, model: str, dataset: str, metric: str) -> List[Tuple[float, float]]:
        try:
            points = self.methods[method][model][dataset][metric]
        except KeyError as e:
            raise KeyError(
                f"Missing curve for method={method}, model={model}, dataset={dataset}, metric={metric}"
            ) from e
        if not points:
            raise ValueError(f"Empty curve points for {method}/{model}/{dataset}/{metric}")
        return [(float(x), float(y)) for x, y in points]

    def estimate(self, method: str, model: str, dataset: str, metric: str, x: float) -> float:
        points = self.curve_points(method, model, dataset, metric)
        return interp_piecewise(points, x)


def load_profile_db(path: str) -> ProfileDB:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    if "methods" not in raw:
        raise ValueError("profile JSON must have top-level key 'methods'")

    # Normalize into the expected nested dict.
    methods: Dict[str, Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]] = {}
    for method, models in raw["methods"].items():
        methods[method] = {}
        for model, datasets in models.items():
            methods[method][model] = {}
            for dataset, metrics in datasets.items():
                methods[method][model][dataset] = {}
                for metric, pts in metrics.items():
                    if not isinstance(pts, list):
                        raise ValueError(
                            f"Curve {method}/{model}/{dataset}/{metric} must be a list of [x,y] points"
                        )
                    methods[method][model][dataset][metric] = [(float(x), float(y)) for x, y in pts]

    return ProfileDB(methods=methods)
