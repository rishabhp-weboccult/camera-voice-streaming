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
        # if (new_w, new_h) == (self.input_width, self.input_height):
        #     canvas = resized.astype(np.float32) * (1.0 / 255.0)
        # else:
        #     resized_float = resized.astype(np.float32) * (1.0 / 255.0)
        #     canvas = np.full((self.input_height, self.input_width, 3), 0.44705882, dtype=np.float32)
        #     canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_float
        
        canvas = np.full((self.input_height, self.input_width, 3), 144, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        
        # Convert to float32 and normalize
        img_arr = canvas.astype(np.float32) / 255.0
        
        # Transpose the float array, NOT the original canvas
        img_arr = img_arr.transpose(2, 0, 1) 
        
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
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image file not found: {image_input}")
            image = Image.open(image_input).convert("RGB")
            img_np = np.array(image)
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            img_np = np.array(image)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            raise TypeError("Unsupported image input type. Use file path (str), PIL Image, or NumPy array.")
        
        original_size = (img_np.shape[1], img_np.shape[0])
        input_tensor, ratio, padding = self._preprocess_numpy(img_np)
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



