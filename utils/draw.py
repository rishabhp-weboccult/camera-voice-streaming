import cv2
import numpy as np
from typing import List, Dict, Any
from dataclass.detection import Detection

def _draw_detections_cv2(
    detector,
    frame: np.ndarray,
    detections: List[Detection],
    fps: float = None,
    logic_status: Dict[str, Any] = None,
    timestamp: float = None,
    is_stationary: bool = None,
    required_stop_time: float = 1.0,
    flow_mag: float = 0.0,
    roi_x1: int = None,
    roi_y: int = None,
    roi_x2: int = None,
    height: int = None,
    y_offset: int = 0,
    detection_fps: float = None,
    flow_time: float = None,
    match_time: float = None
) -> np.ndarray:
    """Draws bounding boxes, labels, and safety logic status dashboard onto a BGR frame."""
    
    # 1. Draw detection bounding boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        # Emerald green in BGR: (113, 204, 46)
        color = (113, 204, 46)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det.class_name} {det.confidence:.2%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Ensure label fits inside frame
        text_y = max(y1, text_h + 10)
        cv2.rectangle(frame, (x1, text_y - text_h - 4), (x1 + text_w, text_y + baseline), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 2. Draw Safety HUD Dashboard
    # Draw background panel
    cv2.rectangle(frame, (10, 10 + y_offset), (380, 205 + y_offset), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(frame, (10, 10 + y_offset), (380, 205 + y_offset), (100, 100, 100), 1)
    
    cv2.putText(frame, "SAFETY COMPLIANCE HUD", (20, 30 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (20, 35 + y_offset), (370, 35 + y_offset), (100, 100, 100), 1)
    
    # Row 1: FPS and Video Time
    fps_text = f"Full FPS: {fps:.1f}" if fps is not None else "Full FPS: N/A"
    det_fps_text = f"Det FPS: {detection_fps:.1f}" if detection_fps is not None else "Det FPS: N/A"
    time_text = f"Time: {timestamp:.2f}s" if timestamp is not None else "Time: N/A"
    cv2.putText(frame, fps_text, (20, 55 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, det_fps_text, (135, 55 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, time_text, (250, 55 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Row 2: Vehicle Motion State and Flow Magnitude
    if is_stationary is not None:
        motion_state = "STATIONARY" if is_stationary else "MOVING"
        motion_color = (0, 255, 0) if is_stationary else (0, 255, 255) # Green if stationary, Yellow if moving
        cv2.putText(frame, "Vehicle Status: ", (20, 80 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{motion_state} (Flow: {flow_mag:.3f})", (150, 80 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, motion_color, 1, cv2.LINE_AA)
        
    # Row 3: Stop Sign Safety logic status
    if logic_status is not None:
        if logic_status["alert_fired"]:
            status_text = "VIOLATION: RUN STOP SIGN!"
            status_color = (0, 0, 255) # Red
        elif logic_status["vehicle_stopped"]:
            status_text = "COMPLIED: VEHICLE STOPPED"
            status_color = (0, 255, 0) # Green
        elif logic_status["stop_seen"]:
            duration = logic_status["stop_duration"]
            status_text = f"VISIBLE: {duration:.1f}s / {required_stop_time:.1f}s"
            status_color = (0, 255, 255) # Yellow
        else:
            status_text = "CLEAR: NO STOP SIGN"
            status_color = (150, 150, 150) # Grey
            
        cv2.putText(frame, "Stop Sign: ", (20, 105 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (110, 105 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
        
    # Row 4: Inference Delegate/Provider
    provider_name = detector.session.get_providers()[0]
    if "CUDA" in provider_name:
        display_provider = "CUDA (GPU)"
        provider_color = (113, 204, 46) # Emerald green
    elif "CPU" in provider_name:
        display_provider = "CPU"
        provider_color = (255, 180, 50) # Orange/yellow
    else:
        display_provider = provider_name
        provider_color = (200, 200, 200) # Grey
        
    cv2.putText(frame, "Delegate: ", (20, 130 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, display_provider, (95, 130 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, provider_color, 1, cv2.LINE_AA)

    # Row 5: Prep/Inf/Post average times (in milliseconds)
    avg_prep_ms = detector.avg_preprocess_time * 1000
    avg_inf_ms = detector.avg_inference_time * 1000
    avg_post_ms = detector.avg_postprocess_time * 1000
    latency_text = f"Prep: {avg_prep_ms:.1f}ms | Inf: {avg_inf_ms:.1f}ms | Post: {avg_post_ms:.1f}ms"
    cv2.putText(frame, latency_text, (20, 155 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # Row 6: Flow and Audio Match times (in milliseconds)
    flow_str = f"Flow: {flow_time * 1000:.1f}ms" if flow_time is not None else "Flow: N/A"
    match_str = f"Match: {match_time * 1000:.1f}ms" if match_time is not None else "Match: N/A"
    latency_text_2 = f"{flow_str} | {match_str}"
    cv2.putText(frame, latency_text_2, (20, 175 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    # 3. Draw Optical Flow ROI boundary (thin blue rectangle)
    if roi_x1 is not None and roi_y is not None and roi_x2 is not None and height is not None:
        cv2.rectangle(frame, (roi_x1, roi_y), (roi_x2, height), (255, 0, 0), 1)
        
    return frame
