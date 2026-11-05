"""Hand-set initial coefficients; statistical calibration follows labelled evaluation."""

import json
from pathlib import Path

from scipy.special import expit

from app.config import StrictModel


class Calibrator(StrictModel):
    intercept: float
    weights: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        """Write coefficients to a new destination without replacing prior fits."""
        with path.open("x", encoding="utf-8") as stream:
            stream.write(self.model_dump_json(indent=2))

    def probability(self, features: dict[str, float]) -> float:
        if features.keys() != self.weights.keys():
            raise ValueError("Verifier features must match the calibration coefficients")
        return float(expit(self.intercept + sum(w * features[k] for k, w in self.weights.items())))
