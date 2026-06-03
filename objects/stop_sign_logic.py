from dataclasses import dataclass
from typing import List, Optional

# Import the local Detection dataclass
from dataclass.detection import Detection

class StopSignLogic:
    """
    Implements compliance checking logic for stop signs.
    Tracks whether a stop sign is visible, how long the vehicle has been stationary 
    while the stop sign is visible, and triggers an alert if the vehicle passes 
    without stopping for the required time limit.
    
    Includes robust enhancements for:
      - Temporal dropout grace period: Prevents false positive alerts due to flickering frames.
      - Alert timeout: Automatically clears the alert overlay after a configurable duration.
      - Full resets between stop signs: Clears compliance status once the sign is out of view.
    """
    def __init__(
        self,
        stop_class_name: str = "stop",
        required_stop_time: float = 1.0,
        grace_period: float = 0.5,
        alert_duration: float = 3.0,
    ):
        self.stop_class_name = stop_class_name
        self.required_stop_time = required_stop_time
        self.grace_period = grace_period
        self.alert_duration = alert_duration

        self.stop_seen = False
        self.stop_start: Optional[float] = None
        self.last_seen_time: Optional[float] = None

        self.vehicle_stop_start: Optional[float] = None
        self.vehicle_stopped = False

        self.alert_fired = False
        self.alert_start_time: Optional[float] = None

    def update(
        self,
        timestamp: float,
        detections: List[Detection],
        is_stationary: bool,
    ) -> dict:
        """
        Update stop-sign state.

        Args:
            timestamp: Current video timestamp (seconds).
            detections: List[Detection] for current frame.
            is_stationary: True if vehicle is stationary.

        Returns:
            dict: {
                "stop_seen": bool,
                "stop_duration": float,
                "vehicle_stopped": bool,
                "alert_fired": bool
            }
        """
        # 1. Check if stop sign is directly detected in this frame
        stop_detected = any(
            d.class_name == self.stop_class_name
            for d in detections
        )

        if stop_detected:
            self.last_seen_time = timestamp

        # 2. Determine if stop sign is effectively visible (present or within dropout grace period)
        is_effectively_visible = False
        if self.stop_seen:
            if stop_detected:
                is_effectively_visible = True
            elif self.last_seen_time is not None and (timestamp - self.last_seen_time) <= self.grace_period:
                is_effectively_visible = True

        # 3. Handle state transitions
        if stop_detected and not self.stop_seen:
            # Transition: New stop sign detected
            self.stop_seen = True
            self.stop_start = timestamp
            self.vehicle_stop_start = None
            self.vehicle_stopped = False
            self.alert_fired = False
            self.alert_start_time = None
            is_effectively_visible = True

        if self.stop_seen and is_effectively_visible:
            # Case A: Stop sign is in tracking range
            if is_stationary:
                if self.vehicle_stop_start is None:
                    self.vehicle_stop_start = timestamp
                elif (
                    timestamp - self.vehicle_stop_start
                    >= self.required_stop_time
                ):
                    self.vehicle_stopped = True
            else:
                self.vehicle_stop_start = None

        elif self.stop_seen and not is_effectively_visible:
            # Case B: Stop sign tracking ended (lost for longer than the grace period)
            if not self.vehicle_stopped and not self.alert_fired:
                self.alert_fired = True
                self.alert_start_time = timestamp

            # Reset tracking states for the next stop sign event
            self.stop_seen = False
            self.stop_start = None
            self.vehicle_stop_start = None
            self.last_seen_time = None
            self.vehicle_stopped = False  # Clear compliance status between signs

        # 4. Handle alert timeout duration check
        if self.alert_fired and self.alert_start_time is not None:
            if (timestamp - self.alert_start_time) >= self.alert_duration:
                self.alert_fired = False
                self.alert_start_time = None

        return {
            "stop_seen": self.stop_seen,
            "stop_duration": (
                timestamp - self.stop_start
                if self.stop_seen and self.stop_start is not None
                else 0.0
            ),
            "vehicle_stopped": self.vehicle_stopped,
            "alert_fired": self.alert_fired,
        }
