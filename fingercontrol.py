import cv2
import numpy as np
import pyautogui
import time
import math
import platform

class GestureTracker:
    def __init__(self):
        print(" Gesture Control - Starting...")
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.05
        
        # Camera setup
        self.cap = self.setup_camera()
        if not self.cap:
            print("Camera initialization failed!")
            return
        
        # Reference finger tracking
        self.reference_finger = None
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_count = 0
        self.required_samples = 10
        
        # Gesture control (will be overridden per mode)
        self.action_cooldown = 2  # default, will set per mode
        self.last_action_time = 0
        
        # Movement tracking
        self.position_history = []
        self.history_size = 8
        self.movement_threshold = 35  # pixel threshold
        
        # Tracking stability
        self.stable_position = None
        self.stability_count = 0
        self.stability_threshold = 5
        
        # Hover detection for buttons
        self.hover_counts = {
            'quick': 0,
            'reading': 0,
            'start': 0,
            'stop': 0,
            'prev': 0,
            'next': 0
        }
        self.hover_threshold = 15  # consecutive frames to register a press
        
        # Reading mode action cooldown to prevent repeat
        self.last_read_action_time = 0
        self.read_action_cooldown = 1.0  # seconds between prev/next presses
        
        # State machine
        # States: 'CALIBRATION', 'MODE_SELECTION', 'QUICK_WAIT_START', 'QUICK_RUNNING', 'READING_RUNNING'
        self.state = 'CALIBRATION'
        self.mode = None  # 'quick' or 'reading'
        
        # Face detector to ignore face region
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Message to display current action ("NEXT PAGE" or "BACK PAGE")
        self.current_action_message = ''
        
        print("Initialization complete!")
    
    def setup_camera(self):
        """Setup camera with error handling"""
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"📹 Camera {i} connected")
                    return cap
                cap.release()
        return None
    
    def calibrate_finger(self):
        """Calibrate and register the user's finger (10 samples)"""
        print("\n FINGER CALIBRATION")
        print("Place your index finger in the green box and hold steady")
        print("Press SPACE when ready, ESC to skip")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Draw calibration box
            box_size = 60
            center_x, center_y = w // 2, h // 2
            cv2.rectangle(
                frame,
                (center_x - box_size, center_y - box_size),
                (center_x + box_size, center_y + box_size),
                (0, 255, 0), 3
            )
            
            # Instructions
            cv2.putText(
                frame,
                "Place finger in box - SPACE to calibrate",
                (w // 2 - 180, center_y - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            if self.calibration_count > 0:
                cv2.putText(
                    frame,
                    f"Samples: {self.calibration_count}/{self.required_samples}",
                    (w // 2 - 80, center_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )
            
            cv2.imshow('Gesture Control - Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Sample the calibration area
                cal_region = frame[
                    center_y - box_size // 2 : center_y + box_size // 2,
                    center_x - box_size // 2 : center_x + box_size // 2
                ]
                
                if cal_region.size > 0:
                    hsv_sample = cv2.cvtColor(cal_region, cv2.COLOR_BGR2HSV)
                    mean_hsv = np.mean(hsv_sample.reshape(-1, 3), axis=0)
                    self.calibration_samples.append(mean_hsv)
                    self.calibration_count += 1
                    
                    if self.calibration_count >= self.required_samples:
                        self.finalize_calibration()
                        break
                        
            elif key == 27:  # ESC
                print("Calibration skipped - using default settings")
                self.use_default_calibration()
                break
                
        cv2.destroyAllWindows()
        self.state = 'MODE_SELECTION'
    
    def finalize_calibration(self):
        """Process calibration samples to set skin HSV range"""
        if not self.calibration_samples:
            self.use_default_calibration()
            return
            
        samples_array = np.array(self.calibration_samples)
        mean_hsv = np.mean(samples_array, axis=0)
        std_hsv = np.std(samples_array, axis=0)
        
        h_range = max(15, std_hsv[0] * 2)
        s_range = max(60, std_hsv[1] * 1.5)
        v_range = max(60, std_hsv[2] * 1.5)
        
        self.skin_lower = np.array([
            max(0, mean_hsv[0] - h_range),
            max(0, mean_hsv[1] - s_range),
            max(0, mean_hsv[2] - v_range)
        ])
        
        self.skin_upper = np.array([
            min(179, mean_hsv[0] + h_range),
            255,
            255
        ])
        
        self.is_calibrated = True
        print("Finger calibrated successfully!")
    
    def use_default_calibration(self):
        """Use default skin HSV thresholds if calibration is skipped"""
        self.skin_lower = np.array([0, 30, 60])
        self.skin_upper = np.array([20, 255, 255])
        self.is_calibrated = True
    
    def detect_finger(self, frame):
        """
        Detect fingertip based on calibrated skin color mask,
        ignoring any contours overlapping detected face regions.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        # Detect faces in the frame (grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (1500 < area < 15000):
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.3 < aspect_ratio < 3.0):
                continue
            
            # Check if this contour overlaps any detected face bounding box
            overlap_with_face = False
            for (fx, fy, fw, fh) in faces:
                if (x < fx + fw and x + w > fx and
                    y < fy + fh and y + h > fy):
                    overlap_with_face = True
                    break
            
            if not overlap_with_face:
                valid_contours.append(cnt)
        
        if not valid_contours:
            return None, None
        
        finger_contour = max(valid_contours, key=cv2.contourArea)
        topmost = tuple(finger_contour[finger_contour[:, :, 1].argmin()][0])
        return topmost, finger_contour
    
    def is_position_stable(self, current_pos):
        """Determine if fingertip is effectively stable (little movement)"""
        if not current_pos:
            return False
        
        if self.stable_position is None:
            self.stable_position = current_pos
            self.stability_count = 1
            return False
        
        dist = math.hypot(
            current_pos[0] - self.stable_position[0],
            current_pos[1] - self.stable_position[1]
        )
        
        if dist < 20:
            self.stability_count += 1
            return self.stability_count >= self.stability_threshold
        else:
            self.stable_position = current_pos
            self.stability_count = 1
            return False
    
    def detect_gesture(self, fingertip):
        """
        Detect 'LEFT' or 'RIGHT' swipe gestures, with cooldown enforced.
        Called only in QUICK_RUNNING mode.
        """
        if time.time() - self.last_action_time < self.action_cooldown:
            return None
        
        if not fingertip:
            return None
        
        # Append to trajectory
        self.position_history.append(fingertip)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        if len(self.position_history) < 4:
            return None
        
        if self.is_position_stable(fingertip):
            return None
        
        # Net horizontal displacement
        start = self.position_history[0]
        end = self.position_history[-1]
        net_dx = end[0] - start[0]
        
        # Count per-frame horizontal deltas
        deltas = [self.position_history[i+1][0] - self.position_history[i][0]
                  for i in range(len(self.position_history)-1)]
        count_pos = sum(1 for d in deltas if d > 5)
        count_neg = sum(1 for d in deltas if d < -5)
        
        if net_dx > self.movement_threshold and count_pos > count_neg:
            return "RIGHT"
        if net_dx < -self.movement_threshold and count_neg > count_pos:
            return "LEFT"
        
        return None
    
    def reset_hover_counts(self):
        """Zero out all hover counters"""
        for key in self.hover_counts:
            self.hover_counts[key] = 0
    
    def draw_interface(self, frame, fingertip, finger_contour):
        """Draw overlays depending on current state, including action messages."""
        h, w = frame.shape[:2]
        
        # Always draw fingertip contour if available (for debug)
        if fingertip and finger_contour is not None:
            cv2.drawContours(frame, [finger_contour], -1, (0, 255, 255), 2)
            cv2.circle(frame, fingertip, 8, (0, 255, 0), -1)
        
        # Display current action message at top center if exists
        if self.current_action_message:
            cv2.putText(
                frame,
                self.current_action_message,
                (w // 2 - 100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2
            )
        
        # Draw based on state
        if self.state == 'MODE_SELECTION':
            # Draw mode boxes
            quick_box = (w // 2 - 150, h // 2 - 30, 140, 60)
            reading_box = (w // 2 + 10, h // 2 - 30, 140, 60)

            
            cv2.rectangle(frame,
                          (quick_box[0], quick_box[1]),
                          (quick_box[0] + quick_box[2], quick_box[1] + quick_box[3]),
                          (255, 0, 0), 2)
            cv2.putText(frame, "Fast Mode", (quick_box[0] + 10, quick_box[1] + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


            
            cv2.rectangle(frame,
                          (reading_box[0], reading_box[1]),
                          (reading_box[0] + reading_box[2], reading_box[1] + reading_box[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, "Reading Mode", (reading_box[0] + 5, reading_box[1] + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elif self.state == 'QUICK_WAIT_START':
            # Draw "Start" button at center
            start_w, start_h = 200, 80
            sx = w // 2 - start_w // 2
            sy = h // 2 - start_h // 2
            cv2.rectangle(frame, (sx, sy), (sx + start_w, sy + start_h), (0, 165, 255), -1)
            cv2.putText(frame, "START", (sx + 50, sy + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Draw "Stop" button moved inward from right
            stop_w, stop_h = 150, 60
            bx = w - stop_w - 100  # moved 100px inward
            by = h - stop_h - 10
            cv2.rectangle(frame, (bx, by), (bx + stop_w, by + stop_h), (0, 0, 255), -1)
            cv2.putText(frame, "STOP", (bx + 30, by + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        elif self.state == 'QUICK_RUNNING':
            # Draw "Stop" button moved inward from right
            stop_w, stop_h = 150, 60
            bx = w - stop_w - 100  # moved 100px inward
            by = h - stop_h - 10
            cv2.rectangle(frame, (bx, by), (bx + stop_w, by + stop_h), (0, 0, 255), -1)
            cv2.putText(frame, "STOP", (bx + 30, by + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cooldown_remaining = max(0.0, self.action_cooldown - (time.time() - self.last_action_time))
            if cooldown_remaining > 0.0:
                cv2.putText(frame,
                            f"Cooldown: {cooldown_remaining:.1f}s",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        elif self.state == 'READING_RUNNING':
            # Draw "Previous" and "Next" buttons at center-left and center-right
            btn_w, btn_h = 120, 60
            prev_x = w // 2 - 220
            prev_y = h // 2 - btn_h // 2
            cv2.rectangle(frame, (prev_x, prev_y), (prev_x + btn_w, prev_y + btn_h), (200, 200, 0), -1)
            cv2.putText(frame, "PREV", (prev_x + 10, prev_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            next_x = w // 2 + 100
            next_y = prev_y
            cv2.rectangle(frame, (next_x, next_y), (next_x + btn_w, next_y + btn_h), (200, 200, 0), -1)
            cv2.putText(frame, "NEXT", (next_x + 10, next_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Draw "Stop" button moved inward from right
            stop_w, stop_h = 150, 60
            bx = w - stop_w - 100  # moved 100px inward
            by = h - stop_h - 10
            cv2.rectangle(frame, (bx, by), (bx + stop_w, by + stop_h), (0, 0, 255), -1)
            cv2.putText(frame, "STOP", (bx + 30, by + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Show calibration status at bottom-left for debug
        status = "CALIBRATED" if self.is_calibrated else "NOT CALIBRATED"
        cv2.putText(frame, f"Status: {status}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if self.is_calibrated else (0, 0, 255), 2)
        
        cv2.imshow('Gesture Control', frame)
    
    def run(self):
        """Main loop: handle calibration, mode selection, and both modes' logic."""
        if not self.cap:
            return
        
        # Step 1: Calibration
        self.calibrate_finger()
        
        print("\n🚀 Entering MODE_SELECTION")
        print("Hover left-top for QUICK mode, right-top for READING mode")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detect fingertip each frame
            fingertip, finger_contour = self.detect_finger(frame)
            
            # State machine
            if self.state == 'MODE_SELECTION':
                # Draw mode selection UI
                self.draw_interface(frame, fingertip, finger_contour)
                
                quick_box = (w // 2 - 150, h // 2 - 30, 140, 60)
                reading_box = (w // 2 + 10, h // 2 - 30, 140, 60)

                
                if fingertip:
                    x, y = fingertip
                    # Hover QUICK
                    if quick_box[0] < x < quick_box[0] + quick_box[2] and quick_box[1] < y < quick_box[1] + quick_box[3]:
                        self.hover_counts['quick'] += 1
                    else:
                        self.hover_counts['quick'] = 0
                    
                    # Hover READING
                    if reading_box[0] < x < reading_box[0] + reading_box[2] and reading_box[1] < y < reading_box[1] + reading_box[3]:
                        self.hover_counts['reading'] += 1
                    else:
                        self.hover_counts['reading'] = 0
                else:
                    self.hover_counts['quick'] = 0
                    self.hover_counts['reading'] = 0
                
                # Mode selection triggered?
                if self.hover_counts['quick'] >= self.hover_threshold:
                    self.mode = 'quick'
                    self.state = 'QUICK_WAIT_START'
                    self.reset_hover_counts()
                    self.current_action_message = ''
                    print("🔹 QUICK mode selected")
                    # Set quick-mode parameters
                    self.action_cooldown = 2.0  # 3-second cooldown
                    self.position_history.clear()
                    self.stable_position = None
                    self.stability_count = 0
                    continue
                
                if self.hover_counts['reading'] >= self.hover_threshold:
                    self.mode = 'reading'
                    self.state = 'READING_RUNNING'
                    self.reset_hover_counts()
                    self.current_action_message = ''
                    print("🔹 READING mode selected")
                    # Set reading-mode parameters
                    self.last_read_action_time = 0
                    self.position_history.clear()
                    self.stable_position = None
                    self.stability_count = 0
                    continue
            
            elif self.state == 'QUICK_WAIT_START':
                self.draw_interface(frame, fingertip, finger_contour)
                
                # Draw and check START and STOP
                start_w, start_h = 200, 80
                sx = w // 2 - start_w // 2
                sy = h // 2 - start_h // 2
                start_box = (sx, sy, start_w, start_h)
                
                stop_w, stop_h = 150, 60
                bx = w - stop_w - 100  # moved inward
                by = h - stop_h - 10
                stop_box = (bx, by, stop_w, stop_h)
                
                if fingertip:
                    x, y = fingertip
                    # Hover START
                    if sx < x < sx + start_w and sy < y < sy + start_h:
                        self.hover_counts['start'] += 1
                    else:
                        self.hover_counts['start'] = 0
                    
                    # Hover STOP
                    if bx < x < bx + stop_w and by < y < by + stop_h:
                        self.hover_counts['stop'] += 1
                    else:
                        self.hover_counts['stop'] = 0
                else:
                    self.hover_counts['start'] = 0
                    self.hover_counts['stop'] = 0
                
                # If Start pressed
                if self.hover_counts['start'] >= self.hover_threshold:
                    self.state = 'QUICK_RUNNING'
                    self.reset_hover_counts()
                    self.last_action_time = time.time() - self.action_cooldown  # allow immediate first swipe
                    print("▶️ QUICK mode started (swipe left/right)")
                    continue
                
                # If Stop pressed (return to mode selection)
                if self.hover_counts['stop'] >= self.hover_threshold:
                    self.state = 'MODE_SELECTION'
                    self.mode = None
                    self.reset_hover_counts()
                    self.current_action_message = ''
                    print("Returning to MODE_SELECTION")
                    continue
            
            elif self.state == 'QUICK_RUNNING':
                # Draw and check STOP
                self.draw_interface(frame, fingertip, finger_contour)
                
                stop_w, stop_h = 150, 60
                bx = w - stop_w - 100  # moved inward
                by = h - stop_h - 10
                stop_box = (bx, by, stop_w, stop_h)
                
                if fingertip:
                    x, y = fingertip
                    if bx < x < bx + stop_w and by < y < by + stop_h:
                        self.hover_counts['stop'] += 1
                    else:
                        self.hover_counts['stop'] = 0
                else:
                    self.hover_counts['stop'] = 0
                
                if self.hover_counts['stop'] >= self.hover_threshold:
                    self.state = 'MODE_SELECTION'
                    self.mode = None
                    self.reset_hover_counts()
                    self.current_action_message = ''
                    print("QUICK mode stopped, back to MODE_SELECTION")
                    continue
                
                # Detect swipe gestures
                gesture = self.detect_gesture(fingertip)
                if gesture:
                    if gesture == "RIGHT":
                        pyautogui.press('down')
                        self.current_action_message = "NEXT PAGE"
                        print("➡️ Next page")
                    elif gesture == "LEFT":
                        pyautogui.press('up')
                        self.current_action_message = "BACK PAGE"
                        print("Back page")
                    
                    self.last_action_time = time.time()
                    self.position_history.clear()
                    self.stable_position = None
                    self.stability_count = 0
            
            elif self.state == 'READING_RUNNING':
                # Draw and check PREV, NEXT, STOP
                self.draw_interface(frame, fingertip, finger_contour)
                
                btn_w, btn_h = 120, 60
                prev_x = w // 2 - 220
                prev_y = h // 2 - btn_h // 2
                next_x = w // 2 + 100
                next_y = prev_y
                
                stop_w, stop_h = 150, 60
                bx = w - stop_w - 100  # moved inward
                by = h - stop_h - 10
                
                current_time = time.time()
                
                if fingertip:
                    x, y = fingertip
                    # Hover PREV
                    if prev_x < x < prev_x + btn_w and prev_y < y < prev_y + btn_h:
                        if current_time - self.last_read_action_time >= self.read_action_cooldown:
                            self.hover_counts['prev'] += 1
                        else:
                            self.hover_counts['prev'] = 0
                    else:
                        self.hover_counts['prev'] = 0
                    
                    # Hover NEXT
                    if next_x < x < next_x + btn_w and next_y < y < next_y + btn_h:
                        if current_time - self.last_read_action_time >= self.read_action_cooldown:
                            self.hover_counts['next'] += 1
                        else:
                            self.hover_counts['next'] = 0
                    else:
                        self.hover_counts['next'] = 0
                    
                    # Hover STOP
                    if bx < x < bx + stop_w and by < y < by + stop_h:
                        self.hover_counts['stop'] += 1
                    else:
                        self.hover_counts['stop'] = 0
                else:
                    self.hover_counts['prev'] = 0
                    self.hover_counts['next'] = 0
                    self.hover_counts['stop'] = 0
                
                # Trigger PREV
                if self.hover_counts['prev'] >= self.hover_threshold:
                    pyautogui.press('up')
                    self.current_action_message = "BACK PAGE"
                    print("Reading: Back page")
                    self.last_read_action_time = current_time
                    self.hover_counts['prev'] = 0
                
                # Trigger NEXT
                if self.hover_counts['next'] >= self.hover_threshold:
                    pyautogui.press('down')
                    self.current_action_message = "NEXT PAGE"
                    print("➡️ Reading: Next page")
                    self.last_read_action_time = current_time
                    self.hover_counts['next'] = 0
                
                # Stop and return to mode selection
                if self.hover_counts['stop'] >= self.hover_threshold:
                    self.state = 'MODE_SELECTION'
                    self.mode = None
                    self.reset_hover_counts()
                    self.current_action_message = ''
                    print("READING mode stopped, back to MODE_SELECTION")
                    continue
            
            # Show overlays and update window
            self.draw_interface(frame, fingertip, finger_contour)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Re-calibrate from scratch
                self.calibration_samples.clear()
                self.calibration_count = 0
                self.stable_position = None
                self.stability_count = 0
                self.position_history.clear()
                self.hover_counts = {k: 0 for k in self.hover_counts}
                self.current_action_message = ''
                self.state = 'CALIBRATION'
                self.calibrate_finger()
                print("\n Re-entering MODE_SELECTION")
                continue
        
        self.cleanup()
    
    def cleanup(self):
        """Release camera and destroy windows"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n Gesture control stopped")


if __name__ == "__main__":
    controller = GestureTracker()
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        controller.cleanup()
