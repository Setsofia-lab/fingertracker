import cv2
import numpy as np
import pyautogui
import time
import math
import platform
import sys

class GestureTracker:
    def __init__(self):
        print("🔧 GESTURE CONTROL TROUBLESHOOTER")
        print("=" * 50)
        
        # Disable pyautogui failsafe for testing
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1
        
        # Test system compatibility
        self.test_system_compatibility()
        
        # Camera setup
        self.setup_camera()
        
        # Gesture variables
        self.last_action_time = 0
        self.action_cooldown = 1.0
        self.gesture_buffer = []
        self.buffer_size = 2
        
        # Skin detection (optimized for various skin tones)
        self.skin_lower = np.array([0, 20, 70])
        self.skin_upper = np.array([20, 255, 255])
        
        # Wave detection
        self.wave_positions = []
        self.wave_detection_window = 15
        
        print("\n✅ Troubleshooter initialized successfully!")
        
    def test_system_compatibility(self):
        """Test system and pyautogui compatibility"""
        print(f"\n🖥️  System Info:")
        print(f"   Platform: {platform.system()} {platform.release()}")
        print(f"   Python: {sys.version.split()[0]}")
        
        # Test pyautogui functionality
        print(f"\n🎮 Testing PyAutoGUI:")
        try:
            screen_size = pyautogui.size()
            print(f"   ✅ Screen size detected: {screen_size}")
            
            # Test if we can get mouse position
            mouse_pos = pyautogui.position()
            print(f"   ✅ Mouse position: {mouse_pos}")
            
            # Test key press (safe test)
            print(f"   🔄 Testing key press in 3 seconds...")
            time.sleep(3)
            pyautogui.press('space')
            print(f"   ✅ Space key press sent successfully!")
            
        except Exception as e:
            print(f"   ❌ PyAutoGUI Error: {e}")
            print(f"   💡 Solution: Install with: pip install pyautogui")
            
            # macOS specific fix
            if platform.system() == "Darwin":
                print(f"   🍎 macOS Fix: Grant accessibility permissions:")
                print(f"      System Preferences > Security & Privacy > Accessibility")
                print(f"      Add Python/Terminal to allowed apps")
    
    def setup_camera(self):
        """Setup and test camera"""
        print(f"\n📹 Camera Setup:")
        
        # Try different camera indices
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.cap = cap
                    print(f"   ✅ Camera {i} connected successfully!")
                    print(f"   📐 Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        
        if self.cap is None:
            print(f"   ❌ No cameras found!")
            print(f"   💡 Solutions:")
            print(f"      1. Check camera connection")
            print(f"      2. Close other apps using camera")
            print(f"      3. Try running as administrator")
            return False
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def test_gesture_detection(self):
        """Interactive gesture detection test"""
        if not self.cap:
            return
            
        print(f"\n🎯 GESTURE DETECTION TEST")
        print(f"=" * 30)
        print(f"Instructions:")
        print(f"1. Position yourself 2-3 feet from camera")
        print(f"2. Try pointing LEFT and RIGHT clearly")
        print(f"3. Try waving your hand horizontally")
        print(f"4. Watch the detection feedback")
        print(f"5. Press 'q' to quit, 'space' to test key press")
        print(f"\nStarting in 3 seconds...")
        time.sleep(3)
        
        detection_count = {"RIGHT": 0, "LEFT": 0, "WAVE": 0}
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Test manual key press
                pyautogui.press('space')
                print("🔄 Manual space key pressed!")
            elif key == ord('c'):
                self.calibrate_skin_color_interactive(frame)
            
            # Detect gestures
            hand_contour, hand_mask = self.detect_hand(frame)
            fingertip, palm_center, pointing_direction = self.find_fingertip_and_direction(hand_contour)
            
            # Process gestures
            detected_gesture = None
            
            if pointing_direction in ["LEFT", "RIGHT"]:
                detected_gesture = pointing_direction
            elif self.detect_wave_gesture(fingertip):
                detected_gesture = "WAVE"
            
            # Update detection counts
            if detected_gesture:
                detection_count[detected_gesture] += 1
                print(f"🎯 Detected: {detected_gesture} (Count: {detection_count[detected_gesture]})")
                
                # Test actual key press
                current_time = time.time()
                if current_time - self.last_action_time > self.action_cooldown:
                    if detected_gesture == "RIGHT":
                        pyautogui.press('space')
                        print("   ➡️ SPACE pressed (Next slide)")
                    elif detected_gesture == "LEFT":
                        pyautogui.press('left')
                        print("   ⬅️ LEFT ARROW pressed (Previous slide)")
                    elif detected_gesture == "WAVE":
                        pyautogui.press('escape')
                        print("   👋 ESCAPE pressed (Exit)")
                    
                    self.last_action_time = current_time
            
            # Enhanced visual feedback
            self.draw_debug_info(frame, hand_contour, fingertip, palm_center, 
                               pointing_direction, detected_gesture, hand_mask, detection_count)
            
            cv2.imshow('🔧 GESTURE TROUBLESHOOTER', frame)
        
        print(f"\n📊 Detection Summary:")
        for gesture, count in detection_count.items():
            print(f"   {gesture}: {count} detections")
    
    def detect_hand(self, frame):
        """Improved hand detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Improved noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaussian blur for smoothing
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter and find best hand contour
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 3000:  # Larger minimum area
                    # Check if contour looks hand-like
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.4 < aspect_ratio < 2.5:  # Hand proportions
                        valid_contours.append((contour, area))
            
            if valid_contours:
                # Get largest valid contour
                hand_contour = max(valid_contours, key=lambda x: x[1])[0]
                return hand_contour, skin_mask
        
        return None, skin_mask
    
    def find_fingertip_and_direction(self, hand_contour):
        """Enhanced fingertip detection"""
        if hand_contour is None:
            return None, None, None
        
        # Get hand center
        M = cv2.moments(hand_contour)
        if M["m00"] == 0:
            return None, None, None
        
        palm_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Find convex hull and defects
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(hand_contour, hull)
                
                if defects is not None:
                    # Find fingertip candidates
                    fingertip_candidates = []
                    
                    # Method 1: Furthest point from palm center
                    max_distance = 0
                    best_fingertip = None
                    
                    for point in hand_contour:
                        pt = tuple(point[0])
                        distance = math.sqrt((pt[0] - palm_center[0])**2 + (pt[1] - palm_center[1])**2)
                        if distance > max_distance:
                            max_distance = distance
                            best_fingertip = pt
                    
                    # Method 2: Top-most point (for pointing up)
                    topmost = tuple(hand_contour[hand_contour[:,:,1].argmin()][0])
                    
                    # Choose the best fingertip
                    fingertip = best_fingertip if max_distance > 80 else topmost
                    
                    if fingertip and max_distance > 60:
                        # Calculate pointing direction with better thresholds
                        dx = fingertip[0] - palm_center[0]
                        dy = fingertip[1] - palm_center[1]
                        
                        direction = None
                        # More sensitive horizontal detection
                        if abs(dx) > 50:  # Horizontal pointing threshold
                            if dx > 50:
                                direction = "RIGHT"
                            elif dx < -50:
                                direction = "LEFT"
                        
                        return fingertip, palm_center, direction
            except:
                pass
        
        return None, None, None
    
    def detect_wave_gesture(self, fingertip):
        """Enhanced wave detection"""
        if fingertip is None:
            return False
        
        self.wave_positions.append(fingertip[0])
        
        if len(self.wave_positions) > self.wave_detection_window:
            self.wave_positions.pop(0)
        
        if len(self.wave_positions) < 8:
            return False
        
        # Analyze horizontal movement pattern
        direction_changes = 0
        last_direction = None
        
        for i in range(2, len(self.wave_positions)):
            movement = self.wave_positions[i] - self.wave_positions[i-2]
            
            if abs(movement) > 15:  # Movement threshold
                current_direction = "right" if movement > 0 else "left"
                if last_direction and current_direction != last_direction:
                    direction_changes += 1
                last_direction = current_direction
        
        # Calculate movement range
        if len(self.wave_positions) >= 5:
            movement_range = max(self.wave_positions[-5:]) - min(self.wave_positions[-5:])
            
            # Wave detected if enough direction changes and movement
            if direction_changes >= 2 and movement_range > 80:
                self.wave_positions = []  # Reset
                return True
        
        return False
    
    def calibrate_skin_color_interactive(self, frame):
        """Interactive skin color calibration"""
        h, w = frame.shape[:2]
        center_x, center_y = w//2, h//2
        
        # Draw calibration box
        cv2.rectangle(frame, (center_x-60, center_y-60), (center_x+60, center_y+60), (0, 255, 0), 3)
        cv2.putText(frame, "CALIBRATION - Press SPACE", (center_x-120, center_y-80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Sample HSV from center region
            center_region = frame[center_y-30:center_y+30, center_x-30:center_x+30]
            hsv_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            mean_hsv = np.mean(hsv_region.reshape(-1, 3), axis=0)
            
            # Update skin color range with wider tolerance
            h_tol, s_tol, v_tol = 20, 100, 100
            self.skin_lower = np.array([
                max(0, mean_hsv[0] - h_tol),
                max(0, mean_hsv[1] - s_tol),
                max(0, mean_hsv[2] - v_tol)
            ])
            self.skin_upper = np.array([
                min(179, mean_hsv[0] + h_tol),
                255,
                255
            ])
            
            print(f"🎨 Skin calibrated! HSV: {self.skin_lower} → {self.skin_upper}")
    
    def draw_debug_info(self, frame, hand_contour, fingertip, palm_center, 
                       pointing_direction, detected_gesture, hand_mask, detection_count):
        """Enhanced debug visualization"""
        h, w = frame.shape[:2]
        
        # Draw hand contour
        if hand_contour is not None:
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
            
            # Draw bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(hand_contour)
            cv2.rectangle(frame, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)
            
            # Show contour area
            area = cv2.contourArea(hand_contour)
            cv2.putText(frame, f"Area: {int(area)}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw fingertip and palm
        if fingertip and palm_center:
            cv2.circle(frame, fingertip, 10, (0, 255, 255), -1)
            cv2.circle(frame, palm_center, 8, (255, 0, 255), -1)
            cv2.line(frame, palm_center, fingertip, (0, 255, 0), 3)
            
            # Show distance
            distance = math.sqrt((fingertip[0] - palm_center[0])**2 + (fingertip[1] - palm_center[1])**2)
            cv2.putText(frame, f"Dist: {int(distance)}", (fingertip[0]+15, fingertip[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status information
        y_offset = 30
        
        # Current gesture
        if detected_gesture:
            color = (0, 255, 0)
            cv2.putText(frame, f"GESTURE: {detected_gesture}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "GESTURE: None", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_offset += 40
        
        # Pointing direction
        if pointing_direction:
            cv2.putText(frame, f"DIRECTION: {pointing_direction}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 35
        
        # Detection counts
        for gesture, count in detection_count.items():
            cv2.putText(frame, f"{gesture}: {count}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 25
        
        # Wave positions indicator
        if len(self.wave_positions) > 0:
            cv2.putText(frame, f"Wave buffer: {len(self.wave_positions)}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Cooldown indicator
        current_time = time.time()
        cooldown_remaining = max(0, self.action_cooldown - (current_time - self.last_action_time))
        if cooldown_remaining > 0:
            cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (w-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Controls: 'q'=quit, 'c'=calibrate, SPACE=test key", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show hand mask in corner
        if hand_mask is not None:
            mask_small = cv2.resize(hand_mask, (120, 90))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[h-100:h-10, w-130:w-10] = mask_color
            cv2.putText(frame, "Hand Mask", (w-125, h-105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_comprehensive_test(self):
        """Run all tests"""
        if not self.cap:
            print("❌ Camera not available, cannot run gesture tests")
            return
        
        print(f"\n🚀 STARTING COMPREHENSIVE TEST")
        print(f"This will test:")
        print(f"1. Hand detection accuracy")
        print(f"2. Gesture recognition")
        print(f"3. Key press functionality")
        print(f"4. Real-time performance")
        
        input("\nPress Enter to start (make sure your presentation is ready)...")
        
        self.test_gesture_detection()
        
        print(f"\n✅ Test completed!")
        print(f"\n🎯 Next Steps:")
        print(f"1. If gestures were detected but slides didn't change:")
        print(f"   - Make sure your presentation app is in focus")
        print(f"   - Try clicking on the presentation window first")
        print(f"   - Check if presentation is in slideshow mode")
        print(f"2. If hand detection was poor:")
        print(f"   - Press 'c' to calibrate skin color")
        print(f"   - Improve lighting conditions")
        print(f"   - Try different hand positions")
        print(f"3. If gestures weren't detected:")
        print(f"   - Make bigger, clearer hand movements")
        print(f"   - Point more horizontally (left/right)")
        print(f"   - Wave more dramatically")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    troubleshooter = GestureTracker()
    try:
        troubleshooter.run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
    finally:
        troubleshooter.cleanup()
        print("🔧 Troubleshooter closed")