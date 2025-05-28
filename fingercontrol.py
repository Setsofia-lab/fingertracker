import cv2
import numpy as np
import pyautogui
import time
import math
import platform

class GestureTracker:
    def __init__(self):
        print("🎯 Gesture Control - Starting...")
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.05
        
        # Camera setup
        self.cap = self.setup_camera()
        if not self.cap:
            print("❌ Camera initialization failed!")
            return
        
        # Reference finger tracking
        self.reference_finger = None
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_count = 0
        self.required_samples = 10
        
        # Gesture control
        self.last_action_time = 0
        self.action_cooldown = 1.5  # Increased cooldown
        
        # Movement tracking
        self.position_history = []
        self.history_size = 8
        self.movement_threshold = 35  # Reduced sensitivity
        
        # Wave detection
        self.wave_positions = []
        self.wave_window = 12
        
        # Tracking stability
        self.stable_position = None
        self.stability_count = 0
        self.stability_threshold = 5
        
        print("✅ Initialization complete!")
        
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
        """Calibrate and register the user's finger"""
        print("\n🖐️ FINGER CALIBRATION")
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
            center_x, center_y = w//2, h//2
            cv2.rectangle(frame, 
                         (center_x - box_size, center_y - box_size),
                         (center_x + box_size, center_y + box_size),
                         (0, 255, 0), 3)
            
            # Instructions
            cv2.putText(frame, "Place finger in box - SPACE to calibrate", 
                       (w//2 - 180, center_y - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.calibration_count > 0:
                cv2.putText(frame, f"Samples: {self.calibration_count}/{self.required_samples}", 
                           (w//2 - 80, center_y + 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Gesture Control - Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Sample the calibration area
                cal_region = frame[center_y - box_size//2:center_y + box_size//2,
                                 center_x - box_size//2:center_x + box_size//2]
                
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
    
    def finalize_calibration(self):
        """Process calibration samples"""
        if not self.calibration_samples:
            self.use_default_calibration()
            return
            
        # Calculate mean and standard deviation
        samples_array = np.array(self.calibration_samples)
        mean_hsv = np.mean(samples_array, axis=0)
        std_hsv = np.std(samples_array, axis=0)
        
        # Create adaptive thresholds
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
        print(f"✅ Finger calibrated successfully!")
    
    def use_default_calibration(self):
        """Use default skin detection values"""
        self.skin_lower = np.array([0, 30, 60])
        self.skin_upper = np.array([20, 255, 255])
        self.is_calibrated = True
    
    def detect_finger(self, frame):
        """Detect and track the calibrated finger"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Filter valid finger contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1500 < area < 15000:  # Finger-sized area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:  # Reasonable proportions
                    valid_contours.append(contour)
        
        if not valid_contours:
            return None, None
        
        # Get the largest valid contour (assumed to be the finger)
        finger_contour = max(valid_contours, key=cv2.contourArea)
        
        # Find fingertip (topmost point)
        topmost = tuple(finger_contour[finger_contour[:,:,1].argmin()][0])
        
        return topmost, finger_contour
    
    def is_position_stable(self, current_pos):
        """Check if finger position is stable"""
        if not current_pos:
            return False
            
        if self.stable_position is None:
            self.stable_position = current_pos
            self.stability_count = 1
            return False
        
        # Calculate distance from stable position
        distance = math.sqrt((current_pos[0] - self.stable_position[0])**2 + 
                           (current_pos[1] - self.stable_position[1])**2)
        
        if distance < 20:  # Position is stable
            self.stability_count += 1
            return self.stability_count >= self.stability_threshold
        else:
            # Position changed significantly
            self.stable_position = current_pos
            self.stability_count = 1
            return False
    
    def detect_gesture(self, fingertip):
        """Detect gestures based on fingertip movement"""
        if not fingertip:
            return None
        
        # Add to position history
        self.position_history.append(fingertip)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        if len(self.position_history) < 4:
            return None
        
        # Check if position is stable (no gesture)
        if self.is_position_stable(fingertip):
            return None
        
        # Calculate movement vector
        start_pos = self.position_history[0]
        end_pos = self.position_history[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Check for directional gestures
        if abs(dx) > self.movement_threshold and abs(dx) > abs(dy) * 1.5:
            if dx > 0:
                return "RIGHT"
            else:
                return "LEFT"
        
        # Check for wave gesture
        if self.detect_wave():
            return "WAVE"
        
        return None
    
    def detect_wave(self):
        """Improved wave detection"""
        if len(self.position_history) < 6:
            return False
        
        # Add current position to wave tracking
        current_x = self.position_history[-1][0]
        self.wave_positions.append(current_x)
        
        if len(self.wave_positions) > self.wave_window:
            self.wave_positions.pop(0)
        
        if len(self.wave_positions) < 8:
            return False
        
        # Analyze for wave pattern (back and forth movement)
        direction_changes = 0
        last_direction = None
        
        for i in range(2, len(self.wave_positions)):
            movement = self.wave_positions[i] - self.wave_positions[i-2]
            
            if abs(movement) > 25:  # Significant movement
                current_direction = "right" if movement > 0 else "left"
                if last_direction and current_direction != last_direction:
                    direction_changes += 1
                last_direction = current_direction
        
        # Check movement range
        movement_range = max(self.wave_positions) - min(self.wave_positions)
        
        # Wave detected if multiple direction changes and good range
        if direction_changes >= 2 and movement_range > 80:
            self.wave_positions = []  # Reset
            return True
        
        return False
    
    def execute_gesture_action(self, gesture):
        """Execute keyboard action for detected gesture"""
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        if gesture == "RIGHT":
            pyautogui.press('right')
            print("➡️ Next slide")
        elif gesture == "LEFT":
            pyautogui.press('left')
            print("⬅️ Previous slide")
        elif gesture == "WAVE":
            pyautogui.press('escape')
            print("👋 Exit presentation")
        
        self.last_action_time = current_time
        self.position_history = []  # Reset movement history
        return True
    
    def draw_interface(self, frame, fingertip, gesture, finger_contour):
        """Draw minimal interface"""
        h, w = frame.shape[:2]
        
        # Draw finger tracking
        if fingertip and finger_contour is not None:
            cv2.drawContours(frame, [finger_contour], -1, (0, 255, 255), 2)
            cv2.circle(frame, fingertip, 8, (0, 255, 0), -1)
        
        # Show current gesture
        if gesture:
            cv2.putText(frame, f"GESTURE: {gesture}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show calibration status
        status = "CALIBRATED" if self.is_calibrated else "NOT CALIBRATED"
        color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Cooldown indicator
        current_time = time.time()
        cooldown_remaining = max(0, self.action_cooldown - (current_time - self.last_action_time))
        if cooldown_remaining > 0:
            cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (w-200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        # Controls
        cv2.putText(frame, "Q: Quit | R: Recalibrate", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main control loop"""
        if not self.cap:
            return
        
        # Initial calibration
        self.calibrate_finger()
        
        print("\n🚀 Gesture control active!")
        print("Gestures: Point LEFT/RIGHT for slides, WAVE to exit")
        print("Press 'Q' to quit, 'R' to recalibrate")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.calibrate_finger()
                continue
            
            # Detect finger and gestures
            fingertip, finger_contour = self.detect_finger(frame)
            gesture = self.detect_gesture(fingertip)
            
            # Execute gesture action
            if gesture:
                self.execute_gesture_action(gesture)
            
            # Draw interface
            self.draw_interface(frame, fingertip, gesture, finger_contour)
            
            cv2.imshow('Gesture Control', frame)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Gesture control stopped")

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