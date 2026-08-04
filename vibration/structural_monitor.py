import numpy as np

class SW420VibrationMonitor:
    """
    SW-420 shock vibration level evaluator.
    Returns shock level (0, 1, 2) and shock alert status based on pulse count.
    """
    def __init__(self):
        pass

    def evaluate(self, pulse_count):
        pulse_count = float(pulse_count)
        if pulse_count >= 50.0:
            return {"shock_level": 2, "shock_alert": True}
        elif pulse_count >= 10.0:
            return {"shock_level": 1, "shock_alert": True}
        else:
            return {"shock_level": 0, "shock_alert": False}


class UltrasonicDisplacementModel:
    """
    Geomechanical rate model using ultrasonic distance sensor telemetry.
    Tracks distance and calculates velocity & acceleration to identify blockages or structural collapse risks.
    """
    def __init__(self):
        # Dictionary mapping node_id to a list of (distance, timestamp) readings
        self.history = {}

    def add_reading(self, distance, timestamp, node_id="default"):
        distance = float(distance)
        timestamp = float(timestamp)
        if node_id not in self.history:
            self.history[node_id] = []
        self.history[node_id].append((distance, timestamp))
        # Keep only the last 10 readings for calculating rolling metrics
        if len(self.history[node_id]) > 10:
            self.history[node_id].pop(0)

    def evaluate(self, node_id="default"):
        hist = self.history.get(node_id, [])
        if len(hist) < 2:
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "blockage_detected": False,
                "collapse_imminent": False
            }

        # Calculate metrics using the last two readings
        d2, t2 = hist[-1]
        d1, t1 = hist[-2]
        
        dt = t2 - t1
        if dt <= 0:
            dt = 2.0  # Fallback to standard tick duration

        # Detect instantaneous drop (e.g. human obstacle passing) as a blockage
        if (d2 - d1) < -0.8:
            return {
                "velocity": 0.0,
                "acceleration": 0.0,
                "blockage_detected": True,
                "collapse_imminent": False
            }

        # Compute displacement velocity (negative indicates wall moving in/closer)
        velocity = (d2 - d1) / dt

        # Compute displacement acceleration if we have at least 3 readings
        acceleration = 0.0
        if len(hist) >= 3:
            d0, t0 = hist[-3]
            dt_prev = t1 - t0
            if dt_prev <= 0:
                dt_prev = 2.0
            
            # Avoid using previous velocity if the previous step was a blockage
            if (d1 - d0) >= -0.8:
                v_prev = (d1 - d0) / dt_prev
                acceleration = (velocity - v_prev) / dt

        # Collapse is imminent if moving inwards at a significant speed and accelerating
        collapse_imminent = (velocity < -0.05) and (acceleration < 0.0)

        return {
            "velocity": float(velocity),
            "acceleration": float(acceleration),
            "blockage_detected": False,
            "collapse_imminent": bool(collapse_imminent)
        }
