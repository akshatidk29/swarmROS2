import cv2
import numpy as np
from cv_bridge import CvBridge

class CameraProcessor:
    def __init__(self, logger):
        self._log = logger
        self.bridge = CvBridge()
        
        # HSV ranges for RED, GREEN, BLUE objects and bins
        # We will use simple colors since Gazebo lighting can vary.
        self.color_ranges = {
            1: [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))], # RED
            2: [(np.array([40, 50, 50]), np.array([80, 255, 255]))],  # GREEN
            3: [(np.array([100, 50, 50]), np.array([140, 255, 255]))] # BLUE
        }

    def process_image(self, msg, depth_msg=None):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Using depth image if provided
            depth_image = None
            if depth_msg:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except Exception as e:
            self._log(f"Camera CV error: {e}")
            return {}

        obj_detections = {}
        bin_detections = {}
        height, width, _ = cv_image.shape
        center_x = width // 2

        for cid, ranges in self.color_ranges.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
                
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_obj_area = 0
            best_obj_cx = -1
            best_obj_cy = -1
            
            best_bin_area = 0
            best_bin_cx = -1
            best_bin_cy = -1
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / max(h, 1)
                    is_bin = aspect_ratio > 1.5
                    
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        if is_bin and area > best_bin_area:
                            best_bin_area = area
                            best_bin_cx = cx
                            best_bin_cy = cy
                        elif not is_bin and area > best_obj_area:
                            best_obj_area = area
                            best_obj_cx = cx
                            best_obj_cy = cy

            def compute_det(best_area, best_cx, best_cy):
                if best_cx < center_x - 100:
                    target_dir = 1
                elif best_cx > center_x + 100:
                    target_dir = 2
                else:
                    target_dir = 0
                
                dist = 9.9
                if depth_image is not None and 0 <= best_cy < height and 0 <= best_cx < width:
                    dist = depth_image[best_cy, best_cx]
                    if np.isnan(dist) or np.isinf(dist):
                        dist = 9.9
                
                is_close = False
                if depth_image is not None:
                    if dist < 0.25:
                        is_close = True
                else:
                    if best_area > 45000:
                        is_close = True
                return (target_dir, is_close, best_area, dist)

            if best_obj_area > 0:
                obj_detections[cid] = compute_det(best_obj_area, best_obj_cx, best_obj_cy)
            if best_bin_area > 0:
                bin_detections[cid] = compute_det(best_bin_area, best_bin_cx, best_bin_cy)

        return obj_detections, bin_detections
