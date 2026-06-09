import json
import logging
import os
import ast
import time
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from typing import List, Dict, Any, Tuple, Union

# Import dataclass from our local module
from dataclass.detection import Detection
from objects.stop_sign_logic import StopSignLogic

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class YOLOv8Detector:
    """
    A standalone ONNX Runtime inference pipeline for YOLOv8/YOLO11 object detection models.
    Supports letterboxing, multi-class non-maximum suppression (NMS), configuration loading, 
    and handles CPU or GPU (CUDA) execution contexts gracefully.
    """
    
    def __init__(self, config_path_or_dict: Union[str, Dict[str, Any]]):
        """
        Initializes the YOLOv8 Detector with configuration options.
        
        Args:
            config_path_or_dict: Path to a JSON configuration file, or a dictionary containing configurations.
        """
        # Load configuration
        if isinstance(config_path_or_dict, str):
            logger.info(f"Loading configuration from file: {config_path_or_dict}")
            with open(config_path_or_dict, 'r') as f:
                self.config = json.load(f)
        elif isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            raise TypeError("config_path_or_dict must be a file path (str) or a dictionary")
            
        # Parse configs with sensible defaults from nested model sub-section
        model_cfg = self.config.get("model", {})
        self.model_path = model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("model_path is required in the configuration.")
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.model_path}")
            
        self.conf_threshold = float(model_cfg.get("conf_threshold", 0.25))
        self.iou_threshold = float(model_cfg.get("iou_threshold", 0.45))
        # Select execution providers based on configuration or auto-detection
        config_providers = model_cfg.get("providers")
        if config_providers:
            if isinstance(config_providers, str):
                config_providers = [config_providers]
                
            available_providers = ort.get_available_providers()
            logger.info(f"System available ONNX providers: {available_providers}")
            
            # Filter configured providers against available ones
            filtered = [p for p in config_providers if p in available_providers]
            if not filtered:
                logger.warning(
                    f"None of the specified providers {config_providers} are available on this system. "
                    "Falling back to ['CPUExecutionProvider']."
                )
                self.providers = ["CPUExecutionProvider"]
            else:
                self.providers = filtered
                logger.info(f"Active execution providers: {self.providers}")
        else:
            self.device = model_cfg.get("device", "auto")
            self.providers = self._select_providers(self.device)

        
        # Instantiate the session
        logger.info(f"Initializing ONNX Runtime Session with model: {self.model_path}")
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        
        # Print actual execution provider used
        logger.info(f"ONNX session initialized. Active provider: {self.session.get_providers()[0]}")
        
        # Cache session input details
        input_details = self.session.get_inputs()[0]
        self.input_name = input_details.name
        input_shape = input_details.shape
        
        # Parse input resolution (check config first, then handle static vs dynamic shapes)
        image_size = model_cfg.get("image_size")
        if isinstance(image_size, list) and len(image_size) == 2:
            self.input_height = image_size[0]
            self.input_width = image_size[1]
            logger.info(f"Model input layer: override from config image_size: {self.input_width}x{self.input_height}")
        else:
            self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
            self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 640
            logger.info(f"Model input layer: '{self.input_name}' with resolution: {self.input_width}x{self.input_height}")
        
        # Load class names mapping
        self.classes = self._load_classes()

        # Latency tracking metrics
        self._preprocess_latencies = []
        self._inference_latencies = []
        self._postprocess_latencies = []
        self.avg_preprocess_time = 0.0
        self.avg_inference_time = 0.0
        self.avg_postprocess_time = 0.0

    def _select_providers(self, device: str) -> List[str]:
        """Determines the appropriate execution providers based on user intent and system features."""
        available = ort.get_available_providers()
        logger.info(f"System available ONNX providers: {available}")
        
        device = device.lower()
        if device in ("gpu", "cuda"):
            if "CUDAExecutionProvider" in available:
                logger.info("Using GPU: CUDAExecutionProvider selected.")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.warning("GPU/CUDA execution provider requested but CUDAExecutionProvider not available on this system. Falling back to CPUExecutionProvider.")
                return ["CPUExecutionProvider"]
        elif device == "cpu":
            logger.info("Using CPU: CPUExecutionProvider selected.")
            return ["CPUExecutionProvider"]
        else:  # "auto" or other
            if "CUDAExecutionProvider" in available:
                logger.info("Auto-select: CUDAExecutionProvider is available and selected.")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.info("Auto-select: CUDAExecutionProvider not available. Fallback to CPUExecutionProvider.")
                return ["CPUExecutionProvider"]

    def _load_classes(self) -> Dict[int, str]:
        """Loads class label mappings from config or falls back to ONNX metadata if unspecified."""
        model_cfg = self.config.get("model", {})
        classes_config = model_cfg.get("classes", {})
        # Map keys to integers
        classes = {int(k): str(v) for k, v in classes_config.items()}
        
        if not classes:
            # If classes empty, try to fetch from custom metadata embedded inside the ONNX model
            meta = self.session.get_modelmeta()
            custom_metadata = meta.custom_metadata_map
            if "names" in custom_metadata:
                try:
                    names_dict = ast.literal_eval(custom_metadata["names"])
                    classes = {int(k): str(v) for k, v in names_dict.items()}
                    logger.info(f"Loaded class name mapping from ONNX model metadata: {classes}")
                except Exception as e:
                    logger.warning(f"Found class names in ONNX metadata, but failed to parse: {e}")
                    
        # If still empty, default to a generic mapping or single default class
        if not classes:
            logger.warning("No class mapping provided in config or ONNX metadata. Defaulting to class index names.")
            
        return classes

    def _preprocess(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Applies aspect-ratio letterboxing, normalization, transposition, 
        and batch expansion to prepare PIL Image for ONNX model.
        
        Returns:
            input_tensor: numpy array of shape (1, 3, height, width).
            ratio: Scale ratio applied to match input resolution.
            padding: Padding offset (pad_left, pad_top) applied.
        """
        original_width, original_height = image.size
        
        # Calculate aspect ratio scaling factor
        r = min(self.input_width / original_width, self.input_height / original_height)
        
        new_width = int(round(original_width * r))
        new_height = int(round(original_height * r))
        
        # Resize with PIL Bilinear filtering
        resized_image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
        # Embed resized image onto padding canvas
        padded_image = Image.new("RGB", (self.input_width, self.input_height), (114, 114, 114))
        pad_left = (self.input_width - new_width) // 2
        pad_top = (self.input_height - new_height) // 2
        padded_image.paste(resized_image, (pad_left, pad_top))
        
        # Convert image to numpy array, scale values to [0, 1]
        img_arr = np.array(padded_image, dtype=np.float32) / 255.0
        
        # Transpose from HWC to CHW format
        img_arr = img_arr.transpose(2, 0, 1)
        
        # Add batch dimension: [1, 3, H, W]
        input_tensor = np.expand_dims(img_arr, axis=0)
        
        return input_tensor, r, (pad_left, pad_top)

    def _preprocess_numpy(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Fast OpenCV-based letterboxing and preprocessing for NumPy arrays.
        Avoids PIL conversion overhead entirely.
        """
        h, w = img.shape[:2]
        
        # Calculate aspect ratio scaling factor
        r = min(self.input_width / w, self.input_height / h)
        new_w = int(round(w * r))
        new_h = int(round(h * r))
        
        # Resize image using OpenCV
        if (new_w, new_h) != (w, h):
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img
            
        # Pad the resized image to the model input size
        pad_left = (self.input_width - new_w) // 2
        pad_top = (self.input_height - new_h) // 2
        
        # Optimization: scale the smaller resized image first to perform fewer operations
        # Optimization: if no padding is required, skip allocation and copy steps entirely
        if (new_w, new_h) == (self.input_width, self.input_height):
            canvas = resized.astype(np.float32) * (1.0 / 255.0)
        else:
            resized_float = resized.astype(np.float32) * (1.0 / 255.0)
            canvas = np.full((self.input_height, self.input_width, 3), 0.44705882, dtype=np.float32)
            canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_float
        
        # Transpose from HWC to CHW format: [3, H, W]
        img_arr = canvas.transpose(2, 0, 1)
        
        # Add batch dimension: [1, 3, H, W]
        input_tensor = np.expand_dims(img_arr, axis=0)
        
        return input_tensor, r, (pad_left, pad_top)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Vectorized Non-Maximum Suppression (NMS) in NumPy.
        
        Args:
            boxes: Bounding boxes as numpy array of shape (N, 4) in [x1, y1, x2, y2] format.
            scores: Box confidence scores of shape (N,).
            iou_threshold: Intersection over Union threshold.
            
        Returns:
            keep: List of indices to keep.
        """
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            # Compute intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # Width and height of overlap
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            # Compute IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-15)
            
            # Select boxes that have low overlap with the current box
            remaining_indices = np.where(iou <= iou_threshold)[0]
            order = order[remaining_indices + 1]
            
        return keep

    def _postprocess(self, output: np.ndarray, original_size: Tuple[int, int], ratio: float, padding: Tuple[int, int]) -> List[Detection]:
        """
        Converts raw network outputs to Detection dataclasses using scaling & NMS.
        
        Args:
            output: The model output tensor, expected shape [1, 4 + C, 8400].
            original_size: (width, height) of the original unscaled image.
            ratio: Scaling factor used during letterbox preprocessing.
            padding: Padding offset (pad_left, pad_top) used during letterbox preprocessing.
            
        Returns:
            detections: List of parsed Detection instances.
        """
        # Squeeze batch dimension and transpose: shape [num_anchors, 4 + num_classes]
        predictions = np.squeeze(output, axis=0).T
        
        # YOLOv8 format: boxes = predictions[:, :4], scores = predictions[:, 4:]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Vectorized extraction of highest confidence class
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter predictions below confidence threshold
        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert cx, cy, w, h (center format) to x1, y1, x2, y2 (corner format)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # Perform multi-class NMS by applying a separation offset for different classes
        max_coordinate = boxes_xyxy.max()
        offsets = class_ids * (max_coordinate + 1.0)
        boxes_for_nms = boxes_xyxy + offsets[:, None]
        
        keep_indices = self._nms(boxes_for_nms, confidences, self.iou_threshold)
        
        # Build Detection objects mapped back to original image space
        final_detections = []
        orig_width, orig_height = original_size
        pad_left, pad_top = padding
        
        for idx in keep_indices:
            box = boxes_xyxy[idx]
            
            # Map back to original coordinate system by reversing padding and scale
            x1_orig = (box[0] - pad_left) / ratio
            y1_orig = (box[1] - pad_top) / ratio
            x2_orig = (box[2] - pad_left) / ratio
            y2_orig = (box[3] - pad_top) / ratio
            
            # Clip coordinates to original image size boundary
            x1_orig = max(0.0, min(x1_orig, orig_width))
            y1_orig = max(0.0, min(y1_orig, orig_height))
            x2_orig = max(0.0, min(x2_orig, orig_width))
            y2_orig = max(0.0, min(y2_orig, orig_height))
            
            class_id = int(class_ids[idx])
            class_name = self.classes.get(class_id, f"class_{class_id}")
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidences[idx]),
                box=[float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)]
            )
            final_detections.append(detection)
            
        return final_detections

    def predict(self, image_input: Union[str, Image.Image, np.ndarray]) -> List[Detection]:
        """
        Runs prediction pipeline on an input image.
        
        Args:
            image_input: Can be a file path (str), PIL.Image, or numpy.ndarray (RGB format).
            
        Returns:
            detections: List of Detection instances.
        """
        t_prep_start = time.perf_counter()
        if isinstance(image_input, np.ndarray):
            # Fast-path for NumPy arrays (OpenCV frames)
            original_size = (image_input.shape[1], image_input.shape[0])
            input_tensor, ratio, padding = self._preprocess_numpy(image_input)
        else:
            # Fallback for PIL Image or file path
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Input image file not found: {image_input}")
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise TypeError("Unsupported image input type. Use file path (str), PIL Image, or NumPy array.")
            original_size = image.size
            input_tensor, ratio, padding = self._preprocess(image)
        t_prep_end = time.perf_counter()
            
        # Inference execution
        t_inf_start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        t_inf_end = time.perf_counter()
        
        # Postprocessing: decodes predictions, runs NMS, maps to original size
        t_post_start = time.perf_counter()
        detections = self._postprocess(outputs[0], original_size, ratio, padding)
        t_post_end = time.perf_counter()
        
        # Record latencies
        prep_latency = t_prep_end - t_prep_start
        inf_latency = t_inf_end - t_inf_start
        post_latency = t_post_end - t_post_start
        
        self._preprocess_latencies.append(prep_latency)
        self._inference_latencies.append(inf_latency)
        self._postprocess_latencies.append(post_latency)
        
        if len(self._preprocess_latencies) > 15:
            self._preprocess_latencies.pop(0)
        if len(self._inference_latencies) > 15:
            self._inference_latencies.pop(0)
        if len(self._postprocess_latencies) > 15:
            self._postprocess_latencies.pop(0)
            
        self.avg_preprocess_time = sum(self._preprocess_latencies) / len(self._preprocess_latencies)
        self.avg_inference_time = sum(self._inference_latencies) / len(self._inference_latencies)
        self.avg_postprocess_time = sum(self._postprocess_latencies) / len(self._postprocess_latencies)
        
        return detections

    def _draw_detections_cv2(
        self, 
        frame: np.ndarray, 
        detections: List[Detection], 
        fps: float = None,
        logic_status: dict = None,
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
        import cv2
        
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
        provider_name = self.session.get_providers()[0]
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
        avg_prep_ms = self.avg_preprocess_time * 1000
        avg_inf_ms = self.avg_inference_time * 1000
        avg_post_ms = self.avg_postprocess_time * 1000
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


