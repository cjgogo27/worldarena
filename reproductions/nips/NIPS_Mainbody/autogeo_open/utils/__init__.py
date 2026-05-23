"""
Utils module for AutoGeo
"""

import json
import time
from typing import Dict, Any, List
from dataclasses import asdict


class Logger:
    """Simple logger for debugging"""

    def __init__(self, name: str = "AutoGeo", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(entry)
        if self.verbose:
            print(entry)

    def info(self, message: str):
        self.log(message, "INFO")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def error(self, message: str):
        self.log(message, "ERROR")

    def debug(self, message: str):
        self.log(message, "DEBUG")

    def get_logs(self) -> List[str]:
        return self.logs

    def save_logs(self, filepath: str):
        with open(filepath, "w") as f:
            f.write("\n".join(self.logs))


class ResultSerializer:
    """Serialize results to various formats"""

    @staticmethod
    def to_json(data: Dict[str, Any], indent: int = 2) -> str:
        return json.dumps(data, indent=indent, default=str)

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def to_csv_row(data: Dict[str, Any]) -> str:
        return ",".join(str(v) for v in data.values())


class DistanceCalculator:
    """Calculate distances between geographic coordinates"""

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in km"""
        import math

        R = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def calculate_error(predicted: Dict[str, float], actual: Dict[str, float]) -> float:
        """Calculate distance error in km"""
        return DistanceCalculator.haversine(
            predicted.get("lat", 0),
            predicted.get("lng", 0),
            actual.get("lat", 0),
            actual.get("lng", 0),
        )


class MetricsCalculator:
    """Calculate evaluation metrics for geolocation"""

    @staticmethod
    def recall_at_k(predictions: List[Dict[str, Any]], k_km: float) -> float:
        """Calculate Recall@K"""
        if not predictions:
            return 0.0

        correct = 0
        for pred in predictions:
            if pred.get("error_km", float("inf")) <= k_km:
                correct += 1

        return correct / len(predictions)

    @staticmethod
    def median_error(predictions: List[Dict[str, Any]]) -> float:
        """Calculate median distance error"""
        if not predictions:
            return float("inf")

        errors = sorted([p.get("error_km", float("inf")) for p in predictions])
        n = len(errors)
        if n % 2 == 0:
            return (errors[n // 2 - 1] + errors[n // 2]) / 2
        return errors[n // 2]

    @staticmethod
    def geo_score(
        predictions: List[Dict[str, Any]], max_distance: float = 2500
    ) -> float:
        """Calculate GeoGuessr-style score"""
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            error = pred.get("error_km", max_distance)
            score = max(0, 5000 * (1 - error / max_distance))
            scores.append(score)

        return sum(scores) / len(scores)

    @staticmethod
    def consistency_score(predictions: List[Dict[str, Any]]) -> float:
        """Calculate consistency score for multi-image predictions"""
        if len(predictions) < 2:
            return 1.0

        # Simple consistency: predictions should be in same city/region
        # In practice would use actual geographic clustering
        locations = [p.get("location", "") for p in predictions]
        unique_locations = len(set(locations))

        # Lower is better - less scattered predictions
        return 1.0 / (1.0 + unique_locations)


def format_results(results: Dict[str, Any]) -> str:
    """Format results for display"""
    lines = []
    lines.append("=" * 50)
    lines.append("GEOLOCALIZATION RESULTS")
    lines.append("=" * 50)
    lines.append(f"Total Rounds: {results.get('total_rounds', 'N/A')}")
    lines.append(f"Average Confidence: {results.get('average_confidence', 0):.2f}")
    lines.append("")

    lines.append("Individual Predictions:")
    for pred in results.get("final_predictions", []):
        lines.append(
            f"  - Image {pred.get('image_id', 'N/A')}: {pred.get('hypothesis', 'unknown')} "
            f"(confidence: {pred.get('confidence', 0):.2f})"
        )

    lines.append("")
    joint = results.get("joint_prediction", {})
    lines.append(f"Joint Prediction: {joint.get('location', 'unknown')}")
    lines.append(f"Joint Score: {joint.get('score', 0):.2f}")

    lines.append("=" * 50)
    return "\n".join(lines)
