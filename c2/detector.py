# detector.py
import os
import time
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# small helpers
def _dist(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# 3D model points for solvePnP (generic face model)
_MODEL_3D = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -330.0, -65.0],   # chin
    [-225.0, 170.0, -135.0],# left eye left corner
    [225.0, 170.0, -135.0], # right eye right corner
    [-150.0, -150.0, -125.0],# left mouth corner
    [150.0, -150.0, -125.0] # right mouth corner
], dtype=np.float64)

# MediaPipe FaceMesh indices approximations
_IDX = {
    'nose': 1, 'chin': 152,
    'l_eye_outer': 33, 'r_eye_outer': 263,
    'l_mouth': 61, 'r_mouth': 291
}

class Detector:
    def __init__(
        self,
        cam_source: str | int = 0,
        weights: str = "yolov8n.pt",
        conf_th: float = 0.35,
        snapshot_dir: str = "snapshots",
        snap_min_secs: int = 5,
        head_yaw_deg: float = 25.0,
        head_pitch_deg: float = 20.0,
        mouth_open_ratio: float = 0.55
    ):
        self.cam_source = cam_source
        # VideoCapture - keep open per instance
        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            print(f"[WARN] Could not open camera source: {cam_source}")

        # YOLOv8 model
        self.model = YOLO(weights)

        self.conf_th = conf_th

        # mediapipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=8, refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4)

        # snapshots folder - if a file exists with same name, remove it then make folder
        self.snapshot_dir = snapshot_dir
        if os.path.isfile(self.snapshot_dir):
            os.remove(self.snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.snap_min_secs = max(0, int(snap_min_secs))
        self.last_snapshot_ts: Dict[str, float] = {}
        self.last_snapshot_path: Optional[str] = None
        self.snapshot_count = 0

        # heuristics thresholds
        self.head_yaw_deg = head_yaw_deg
        self.head_pitch_deg = head_pitch_deg
        self.mouth_open_ratio = mouth_open_ratio

        # objects we treat as suspicious by default (COCO labels)
        self.suspicious_labels = {"cell phone", "phone", "book", "laptop"}

    def _maybe_snapshot(self, frame, label: str) -> Optional[str]:
        key = f"{label}"
        now = time.time()
        last = self.last_snapshot_ts.get(key, 0)
        if now - last < self.snap_min_secs:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.snapshot_dir}/{label}_{ts}.jpg".replace(" ", "_")
        success = cv2.imwrite(fname, frame)
        if success:
            self.last_snapshot_ts[key] = now
            self.last_snapshot_path = fname
            self.snapshot_count += 1
            return fname
        return None

    def _head_pose(self, H:int, W:int, landmarks: List[Tuple[int,int]]) -> Tuple[float,float]:
        # Takes face landmarks (full list) and computes approximate yaw/pitch
        try:
            pts2d = np.array([
                landmarks[_IDX['nose']],
                landmarks[_IDX['chin']],
                landmarks[_IDX['l_eye_outer']],
                landmarks[_IDX['r_eye_outer']],
                landmarks[_IDX['l_mouth']],
                landmarks[_IDX['r_mouth']],
            ], dtype=np.float64)
            focal = W
            center = (W/2.0, H/2.0)
            cam_mat = np.array([[focal,0,center[0]],[0,focal,center[1]],[0,0,1]], dtype=np.float64)
            dist_coef = np.zeros((4,1))
            ok, rvec, tvec = cv2.solvePnP(_MODEL_3D, pts2d, cam_mat, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return 0.0, 0.0
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            pitch = math.degrees(math.atan2(-rmat[2,0], sy))
            yaw = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
            return yaw, pitch
        except Exception:
            return 0.0, 0.0

    def process(self) -> Tuple[Optional[Any], List[Dict[str,Any]]]:
        """
        Read a frame, run detections + heuristics.
        Returns:
            annotated_frame (BGR numpy) or None,
            incidents: list of dicts {type,label,conf?,snap?}
        """
        if not self.cap or not self.cap.isOpened():
            return None, []

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, []

        H, W = frame.shape[:2]
        incidents: List[Dict[str,Any]] = []

        # YOLO inference
        results = self.model(frame, conf=self.conf_th, iou=0.45, imgsz=640, verbose=False)
        r = results[0]

        # Gather detections
        names = r.names
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1,y1,x2,y2), conf, cls in zip(xyxy, confs, clss):
                x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
                label = names.get(int(cls), str(int(cls)))
                boxes.append((x1,y1,x2,y2,label,float(conf)))
                # mark incident for suspicious labels
                if label.lower() in self.suspicious_labels and float(conf) >= self.conf_th:
                    snap = self._maybe_snapshot(frame, f"object_{label}")
                    incidents.append({"type":"object","label":label,"conf":float(conf),"snap": (snap or "")})

        # MediaPipe face mesh for pose / mouth
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_res = self.face_mesh.process(rgb)
        face_centers = []
        face_landmarks_px = []
        if mesh_res.multi_face_landmarks:
            for fl in mesh_res.multi_face_landmarks:
                pts = [(int(p.x*W), int(p.y*H)) for p in fl.landmark]
                face_landmarks_px.append(pts)
                cx = int(np.mean([p[0] for p in pts])); cy = int(np.mean([p[1] for p in pts]))
                face_centers.append((cx,cy))

            # check proximity between faces -> possible group discussion
            for i in range(len(face_centers)):
                for j in range(i+1, len(face_centers)):
                    d = _dist(face_centers[i], face_centers[j])
                    if d < max(0.18*W, 120):  # heuristic threshold
                        snap = self._maybe_snapshot(frame, "group_discussion")
                        incidents.append({"type":"proximity","label":"group_discussion","snap": (snap or "")})

            # per-face pose/mouth checks
            for pts in face_landmarks_px:
                # head pose
                yaw, pitch = self._head_pose(H, W, pts)
                statuses = []
                if abs(yaw) > self.head_yaw_deg:
                    statuses.append("looking_side")
                if pitch > self.head_pitch_deg:
                    statuses.append("looking_down")
                # mouth open heuristic: distance between upper & lower lip / eye distance
                upper = pts[13]; lower = pts[14]
                mouth_gap = _dist(upper, lower)
                eye_l = pts[33]; eye_r = pts[263]
                eye_dist = max(_dist(eye_l, eye_r), 1.0)
                mor = mouth_gap / eye_dist
                if mor > self.mouth_open_ratio:
                    statuses.append("talking")
                if statuses:
                    lab = "+".join(statuses)
                    snap = self._maybe_snapshot(frame, lab)
                    incidents.append({"type":"behavior","label":lab,"snap": (snap or "")})
        else:
            # No face detected (possible student left / camera occluded)
            snap = self._maybe_snapshot(frame, "no_face")
            incidents.append({"type":"face","label":"no_face","snap": (snap or "")})

        # MediaPipe hands for hand-near-face heuristic
        hands_res = self.hands.process(rgb)
        hand_points = []
        if hands_res.multi_hand_landmarks:
            for hand in hands_res.multi_hand_landmarks:
                pts = [(int(l.x*W), int(l.y*H)) for l in hand.landmark]
                hand_points.append(pts)
        if face_centers and hand_points:
            for fc in face_centers:
                for hp in hand_points:
                    wrist = hp[0]; tip = hp[8]
                    if _dist(fc, wrist) < max(0.12*W, 100) or _dist(fc, tip) < max(0.12*W, 100):
                        snap = self._maybe_snapshot(frame, "hand_near_face")
                        incidents.append({"type":"gesture","label":"hand_near_face","snap": (snap or "")})

        # simple chit/paper heuristic: bright small rectangle near a hand
        if hand_points:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thr = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
            cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                area = w*h
                if area < 100 or area > 0.05*W*H:
                    continue
                ar = w / (h + 1e-6)
                if 0.4 < ar < 2.5:
                    rect_center = (x+w//2, y+h//2)
                    near_hand = any(_dist(rect_center, hp[8]) < 0.12*W for hp in hand_points)
                    if near_hand:
                        snap = self._maybe_snapshot(frame, "possible_chit")
                        incidents.append({"type":"object","label":"possible_chit","snap": (snap or "")})
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,165,255), 2)

        # Annotate using YOLO's plotting (boxes + labels)
        annotated = r.plot() if r is not None else frame

        # put small summary text
        if incidents:
            cv2.putText(annotated, "ALERT", (12,36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        else:
            cv2.putText(annotated, "OK", (12,36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)

        return annotated, incidents

    def get_last_snapshot(self) -> Optional[str]:
        return self.last_snapshot_path if self.last_snapshot_path and os.path.exists(self.last_snapshot_path) else None

    def release(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
