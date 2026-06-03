from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Detection:
    """
    Represents a single object detection event.
    
    Attributes:
        class_id (int): The index of the predicted class.
        class_name (str): The label/name of the predicted class.
        confidence (float): The detection confidence score between 0.0 and 1.0.
        box (List[float]): Bounding box coordinates [x1, y1, x2, y2] in pixels
                           relative to the original input image.
    """
    class_id: int
    class_name: str
    confidence: float
    box: List[float]

    def to_dict(self) -> dict:
        """Converts the detection instance to a standard dictionary representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": [round(val, 2) for val in self.box]
        }
