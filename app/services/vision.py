import os
import time
import subprocess
import threading
import base64
from collections import deque

import cv2
import numpy as np


class FireMaskTracker:
    """
    คลาสสำหรับติดตามการเปลี่ยนแปลงของพื้นที่ไฟ (Temporal Logic)
    แกะ Logic จาก video_fire_analyzer.py เพื่อหาพื้นที่การขยายตัว (Expansion)
    """

    def __init__(self, history_len):
        self._masks = deque(maxlen=history_len)

    def update(self, current_mask):
        self._masks.append(current_mask.copy())
        if len(self._masks) < 2:
            return None

        # สร้าง Union Mask (รวมพื้นที่ไฟจากเฟรมก่อนๆ ทั้งหมดใน History)
        prev_union = np.zeros_like(current_mask)
        for m in list(self._masks)[:-1]:
            if m.shape == prev_union.shape:
                prev_union = cv2.bitwise_or(prev_union, m)

        # หาพื้นที่ที่ "งอก" ออกมาใหม่ (Current AND NOT Previous_Union)
        expansion = cv2.bitwise_and(current_mask, cv2.bitwise_not(prev_union))

        # Cleanup ด้วย Morphological Open เพื่อลด Noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(expansion, cv2.MORPH_OPEN, kernel)


class HotspotInstance:
    def __init__(self, drone_id, model, input_url, output_url, duration=600):
        self.drone_id = drone_id
        self.model = model
        self.input_url = input_url
        self.output_url = output_url
        self.duration = duration

        # Status & Control
        self.is_running = False
        self.start_time = None

        # Memory Resources
        self.latest_frame = None
        self.latest_results = None
        self.latest_expansion = None
        self.pusher_process = None
        self.lock = threading.Lock()

        # Temporal Logic
        self.tracker = FireMaskTracker(history_len=10)

    def _init_pusher(self, width, height):
        """สร้าง FFmpeg Process สำหรับ Push Stream ไปยัง MediaMTX/RTSP Server"""
        target_fps = "10"
        actual_ai_fps = "2"
        command = [
            "ffmpeg",
            "-y",
            # ขาเข้า
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            actual_ai_fps,
            "-i",
            "-",
            # ขาออก
            "-r",
            target_fps,
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "p1",
            "-delay", "0",
            "-forced-idr", "1",
            "-g",
            target_fps,
            "-cq", "30",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            self.output_url,
        ]
        # ใช้ creationflags=0x08000000 เพื่อซ่อนหน้าต่างดำบน Windows
        return subprocess.Popen(
            command, stdin=subprocess.PIPE, creationflags=0x08000000
        )

    def start(self):
        self.is_running = True
        self.start_time = time.time()
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.input_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"[{self.drone_id}] Failed to open input stream")
            self.is_running = False
            return

        ai_inference_size = 640 
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ratio = ai_inference_size / max(orig_width, orig_height)
        new_w, new_h = int(orig_width * ratio), int(orig_height * ratio)
        # new_w, new_h = (orig_width, orig_height)

        self.pusher_process = self._init_pusher(new_w, new_h)

        try:
            retry_count = 0
            max_retries = 50  # ลองใหม่สักครู่
            while self.is_running:
                # 1. ตรวจสอบอายุ Instance
                if time.time() - self.start_time > self.duration:
                    print(f"[{self.drone_id}] Duration reached, stopping...")
                    break

                for _ in range(4):
                    cap.grab()

                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    if retry_count > max_retries:
                        print(f"[{self.drone_id}] Stream lost permanently.")
                        break
                    time.sleep(0.1)  # พักแป๊บเดียวแล้วลองใหม่
                    continue
                retry_count = 0  # ถ้าอ่านได้ให้รีเซ็ตตัวนับ

                # 2. AI Inference (YOLOv8 Segmentation)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                results = self.model.predict(frame, conf=0.25, verbose=False)

                # 3. จัดการ Mask และ Temporal Logic
                h, w = frame.shape[:2]
                current_mask = np.zeros((h, w), dtype=np.uint8)

                if results[0].masks is not None:
                    for mask_data in results[0].masks.xy:
                        cv2.fillPoly(current_mask, [mask_data.astype(np.int32)], 255)

                expansion = self.tracker.update(current_mask)

                # 4. อัปเดตข้อมูลแชร์ (Thread-safe)
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_results = results
                    self.latest_expansion = (
                        expansion.copy() if expansion is not None else None
                    )

                # 5. วาดภาพสำหรับ Live Stream (Annotated Frame)
                annotated_frame = frame.copy()

                # วาดพื้นที่ไฟปัจจุบัน (สีแดง)
                if results[0].masks is not None:
                    for mask in results[0].masks.xy:
                        cv2.polylines(
                            annotated_frame,
                            [mask.astype(np.int32)],
                            True,
                            (0, 0, 255),
                            2,
                        )

                # วาดพื้นที่ขยายตัว/หัวไฟ (สีเขียว Overlay)
                if expansion is not None and np.any(expansion):
                    overlay = annotated_frame.copy()
                    overlay[expansion == 255] = (0, 255, 0)
                    cv2.addWeighted(
                        annotated_frame, 0.7, overlay, 0.3, 0, dst=annotated_frame
                    )

                # 6. Push Stream ไปยัง Output
                if self.pusher_process.poll() is None:
                    try:
                        self.pusher_process.stdin.write(annotated_frame.tobytes())
                    except BrokenPipeError:
                        break
                else:
                    break

        except Exception as e:
            print(f"Error in {self.drone_id}: {e}")
        finally:
            self.stop()
            cap.release()

    def stop(self):
        self.is_running = False
        if self.pusher_process:
            try:
                self.pusher_process.stdin.close()
                self.pusher_process.terminate()
                self.pusher_process.wait(timeout=2)
            except:
                self.pusher_process.kill()
        print(f"Cleaned up instance: {self.drone_id}")

    def get_snapshot_image(self):
        """ดึงภาพล่าสุดพร้อมข้อมูลการตรวจจับ (สำหรับ Express/Frontend)"""
        with self.lock:
            if self.latest_frame is None:
                return None
            img = self.latest_frame.copy()
            res = self.latest_results
            exp = self.latest_expansion

        # วาดกรอบไฟ (สีแดง)
        if res and res[0].masks is not None:
            for mask in res[0].masks.xy:
                cv2.polylines(img, [mask.astype(np.int32)], True, (0, 0, 255), 3)

        # วาดหัวไฟขยายตัว (สีเขียว)
        if exp is not None and np.any(exp):
            mask_indices = np.where(exp == 255)
            img[mask_indices[0], mask_indices[1]] = [0, 255, 0]

        _, buffer = cv2.imencode(".jpg", img)
        return buffer.tobytes()

    def extend_duration(self, seconds=None):
        """
        รีเซ็ตเวลาเริ่มต้น หรือเพิ่มระยะเวลาการรัน
        """
        if seconds:
            self.duration += seconds
        else:
            # ถ้าไม่ระบุวินาที ให้ถือว่ารีเซ็ตเวลาเริ่มใหม่ (นับถอยหลังใหม่จาก duration เดิม)
            self.start_time = time.time()
        print(f"[{self.drone_id}] Duration extended. New start time: {self.start_time}")

    def get_full_analysis_data(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            # 1. เตรียมภาพ Original (เอาไว้แสดงผลแบบสะอาด)
            _, buffer_orig = cv2.imencode(".jpg", self.latest_frame)
            orig_base64 = base64.b64encode(buffer_orig).decode("utf-8")
            # 2. เตรียมภาพ Detected (ที่มีวาด Mask/Expansion แล้ว)
            # เรียกใช้ logic เดิมที่คุณมีใน get_snapshot_image()
            img_detected = self.latest_frame.copy()
            res = self.latest_results
            exp = self.latest_expansion
            bboxes = []
            if res and res[0].masks is not None:
                # เก็บ BBox ข้อมูลพิกัด (x1, y1, x2, y2)
                bboxes = res[0].boxes.xyxy.tolist() if res[0].boxes is not None else []
                for mask in res[0].masks.xy:
                    cv2.polylines(
                        img_detected, [mask.astype(np.int32)], True, (0, 0, 255), 3
                    )
            if exp is not None and np.any(exp):
                img_detected[exp == 255] = [0, 255, 0]
                
            _, buffer_det = cv2.imencode(".jpg", img_detected)
            det_base64 = base64.b64encode(buffer_det).decode("utf-8")
            return {
                "timestamp": time.time(),
                "drone_id": self.drone_id,
                "original_image": orig_base64,
                "detected_image": det_base64,
                "bboxes": bboxes,
            }
