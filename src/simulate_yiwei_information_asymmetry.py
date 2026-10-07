import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SECTORS = ["northeast", "keelung", "tamsui", "central_west", "south"]
TRUE_SECTOR = 0
LANDING_DAY = 42  # 17 April to 29 May 1895, used only as a historical clock.
HORIZON = 55


SCENARIOS = {
    "historical_asymmetry_stress": {
        "elite_reach": 0.16, "diffusion": 0.045, "signal_precision": 0.30,
        "sharing": 0.18, "declaration_reliance": 0.72,
    },
    "pungdo_informed_prior": {
        "elite_reach": 0.16, "diffusion": 0.045, "signal_precision": 0.30,
        "sharing": 0.18, "declaration_reliance": 0.28,
    },
    "broad_early_warning": {
        "elite_reach": 0.34, "diffusion": 0.105, "signal_precision": 0.30,
        "sharing": 0.55, "declaration_reliance": 0.28,
    },
    "precise_location_intelligence": {
        "elite_reach": 0.16, "diffusion": 0.045, "signal_precision": 0.72,
        "sharing": 0.55, "declaration_reliance": 0.28,
    },
    "integrated_warning_network": {
        "elite_reach": 0.34, "diffusion": 0.105, "signal_precision": 0.72,
        "sharing": 0.78, "declaration_reliance": 0.20,
    },
}


def entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -np.sum(q * np.log(q), axis=1) / np.log(q.shape[1])


def simulate(name, parameters, trajectories=30_000, seed=18950725):
    rng = np.random.default_rng(seed)
    n = trajectories
    posterior = np.tile(np.array([0.20, 0.25, 0.22, 0.18, 0.15]), (n, 1))
    awareness = np.zeros((n, 3), dtype=float)
    awareness[:, 0] = parameters["elite_reach"]
    awareness[:, 1] = 0.24 * parameters["elite_reach"]
    awareness[:, 2] = 0.05 * parameters["elite_reach"]
    readiness = np.zeros(n, dtype=float)
    commitment = rng.normal(-1.15, 0.22, n)
    entered = np.zeros(n, dtype=bool)
    recognized_day = np.full(n, HORIZON + 1, dtype=int)

    for day in range(HORIZON):
        treaty_pressure = 0.015 + 0.018 * (day >= 25)
        jump_probability = 0.012 + 0.055 * (day >= 31)
        jump = (rng.random(n) < jump_probability) * np.maximum(
            0.0, rng.standard_t(df=3, size=n) * 0.18 + 0.42)
        bounded_commitment = np.clip(commitment, -3.0, 3.0)
        commitment += 0.92 * (-bounded_commitment ** 3 + 1.05 * bounded_commitment
                              + treaty_pressure) * 0.08
        commitment += rng.normal(0, 0.045, n) + jump
        commitment = np.clip(commitment, -3.0, 3.0)
        entered |= commitment > 0.45

        preparation = np.clip((day - 25) / 18.0, 0.0, 1.0)
        informative = rng.random(n) < parameters["signal_precision"] * preparation
        observed_sector = rng.integers(0, len(SECTORS), n)
        observed_sector[informative] = TRUE_SECTOR
        likelihood = np.full((n, len(SECTORS)), 0.72)
        likelihood[np.arange(n), observed_sector] = 1.0 + 2.8 * parameters["signal_precision"]
        likelihood = 1.0 + parameters["sharing"] * (likelihood - 1.0)
        posterior *= likelihood
        posterior /= posterior.sum(axis=1, keepdims=True)

        procedural_discount = 1.0 - parameters["declaration_reliance"] * (day < LANDING_DAY)
        warning_intensity = np.clip(
            procedural_discount * (0.45 * entered + 0.55 * posterior[:, TRUE_SECTOR]), 0.0, 1.0)
        current_entropy = entropy(posterior)
        credibility = np.clip(1.0 - 0.62 * current_entropy, 0.20, 1.0)
        elite, local, public = awareness.T
        elite += (0.018 + 0.055 * parameters["sharing"]) * warning_intensity * (1.0 - elite)
        local += parameters["diffusion"] * credibility * (0.86 * elite + 0.14 * local) * (1.0 - local)
        public += parameters["diffusion"] * credibility * (0.58 * local + 0.08 * public) * (1.0 - public)
        awareness += rng.normal(0, 0.0025, awareness.shape)
        awareness = np.clip(awareness, 0.0, 1.0)
        readiness += 0.055 * awareness[:, 1] * posterior[:, TRUE_SECTOR] * (1.0 - readiness)
        readiness = np.clip(readiness, 0.0, 1.0)

        recognized = (posterior[:, TRUE_SECTOR] >= 0.50) & (awareness[:, 1] >= 0.50)
        recognized_day[(recognized_day > HORIZON) & recognized] = day

    correct = posterior[:, TRUE_SECTOR]
    h = entropy(posterior)
    surprise = (1.0 - correct) * (1.0 - awareness[:, 1]) * (1.0 - readiness)
    window = np.maximum(0, np.minimum(recognized_day, LANDING_DAY) - 31)
    unrecognized_at_landing = recognized_day > LANDING_DAY
    return {
        "scenario": name,
        "trajectories": n,
        "correct_sector_posterior_at_horizon": float(np.mean(correct)),
        "normalized_location_entropy": float(np.mean(h)),
        "elite_awareness_index": float(np.mean(awareness[:, 0])),
        "local_organizer_awareness_index": float(np.mean(awareness[:, 1])),
        "public_awareness_index": float(np.mean(awareness[:, 2])),
        "information_stratification_range": float(
            np.mean(np.max(awareness, axis=1) - np.min(awareness, axis=1))),
        "local_readiness_index": float(np.mean(readiness)),
        "unrecognized_at_landing": float(np.mean(unrecognized_at_landing)),
        "mean_information_window_days": float(np.mean(window)),
        "surprise_opportunity_index": float(np.mean(surprise)),
        "warning": "Synthetic mechanism output, not a measured historical population share.",
    }


def main():
    rows = [simulate(name, p, seed=18950725 + i) for i, (name, p) in enumerate(SCENARIOS.items())]
    csv_path = OUT / "乙未战争_信息不对称情景.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": "Bayesian incomplete-information jump with stratified SI warning diffusion",
        "historical_clock": {"t0": "1895-04-17", "landing": "1895-05-29"},
        "hidden_state": ["coercive intent", "landing time", "landing sector"],
        "observer_state": ["sector posterior", "warning reach", "local readiness"],
        "scenarios": rows,
    }
    (OUT / "乙未战争_信息不对称模型摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    by_name = {row["scenario"]: row for row in rows}
    assert by_name["pungdo_informed_prior"]["surprise_opportunity_index"] < by_name["historical_asymmetry_stress"]["surprise_opportunity_index"]
    assert by_name["historical_asymmetry_stress"]["normalized_location_entropy"] > 0.70
    assert by_name["historical_asymmetry_stress"]["public_awareness_index"] < by_name["historical_asymmetry_stress"]["local_organizer_awareness_index"] < by_name["historical_asymmetry_stress"]["elite_awareness_index"]
    assert by_name["precise_location_intelligence"]["normalized_location_entropy"] < by_name["historical_asymmetry_stress"]["normalized_location_entropy"]
    assert by_name["integrated_warning_network"]["surprise_opportunity_index"] < by_name["broad_early_warning"]["surprise_opportunity_index"]
    assert by_name["integrated_warning_network"]["mean_information_window_days"] < by_name["historical_asymmetry_stress"]["mean_information_window_days"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("YIWEI_INFORMATION_MODEL_VERIFICATION: PASS")


if __name__ == "__main__":
    main()
