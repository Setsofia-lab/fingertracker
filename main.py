# import cv2
# import numpy as np
# import pyautogui
# import time

# class InvisibleButtonSystem:
#     def __init__(self):
#         # CAMERA CONNECTION: Connect to default camera (usually webcam)
#         # 0 = default camera, 1 = second camera, etc.
#         self.cap = cv2.VideoCapture(0)
        
#         # Check if camera opened successfully
#         if not self.cap.isOpened():
#             print("ERROR: Cannot access camera!")
#             print("Troubleshooting:")
#             print("1. Make sure no other apps are using the camera")
#             print("2. Try changing camera index (0, 1, 2...)")
#             print("3. Check if camera is properly connected")
#             return
        
#         # Set camera properties for better performance
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         self.cap.set(cv2.CAP_PROP_FPS, 30)
        
#         print(f"Camera connected successfully!")
#         print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
#         print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
        
#         self.last_press_time = 0
#         self.press_cooldown = 1.0  # 1 second between presses
#         self.detection_buffer = []
#         self.buffer_size = 3  # Need 3 consecutive detections
        
#         # Color ranges for different buttons (HSV format)
#         self.colors = {
#             'red': {
#                 'lower': np.array([0, 120, 70]),
#                 'upper': np.array([10, 255, 255]),
#                 'action': self.next_slide
#             },
#             'blue': {
#                 'lower': np.array([100, 150, 50]),
#                 'upper': np.array([140, 255, 255]),
#                 'action': self.previous_slide
#             },
#             'green': {
#                 'lower': np.array([40, 40, 40]),
#                 'upper': np.array([80, 255, 255]),
#                 'action': self.exit_presentation
#             }
#         }
        
#         # Skin color range (adjust these values for your skin tone)
#         self.skin_lower = np.array([0, 20, 70])
#         self.skin_upper = np.array([20, 255, 255])
        
#         print("Invisible Button System Initialized")
#         print("Controls:")
#         print("- Red button: Next slide (Spacebar)")
#         print("- Blue button: Previous slide (Left arrow)")
#         print("- Green button: Exit presentation (Escape)")
#         print("- Press 'q' to quit, 'c' for calibration mode")
    
#     def next_slide(self):
#         """Action for red button"""
#         pyautogui.press('space')
#         print("Next slide!")
    
#     def previous_slide(self):
#         """Action for blue button"""
#         pyautogui.press('left')
#         print("Previous slide!")
    
#     def exit_presentation(self):
#         """Action for green button"""
#         pyautogui.press('escape')
#         print("Exit presentation!")
    
#     def detect_colored_regions(self, frame):
#         """Detect all colored button regions"""
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         button_regions = {}
        
#         for color_name, color_info in self.colors.items():
#             # Create mask for this color
#             mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            
#             # Clean up noise
#             kernel = np.ones((5,5), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
#             # Find contours
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # Filter by area (remove small noise)
#             valid_contours = []
#             for contour in contours:
#                 area = cv2.contourArea(contour)
#                 if area > 500:  # Minimum area threshold
#                     valid_contours.append(contour)
            
#             if valid_contours:
#                 button_regions[color_name] = {
#                     'contours': valid_contours,
#                     'action': color_info['action']
#                 }
        
#         return button_regions
    
#     def detect_hand(self, frame):
#         """Detect hand using skin color segmentation"""
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # Create skin mask
#         skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
#         # Clean up the mask
#         kernel = np.ones((3,3), np.uint8)
#         skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
#         skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
#         # Find hand contour (largest contour)
#         contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if contours:
#             # Get the largest contour (assume it's the hand)
#             hand_contour = max(contours, key=cv2.contourArea)
            
#             # Only consider it a hand if it's large enough
#             if cv2.contourArea(hand_contour) > 1000:
#                 return hand_contour, skin_mask
        
#         return None, skin_mask
    
#     def find_fingertip(self, hand_contour):
#         """Find fingertip as the topmost point of hand contour"""
#         if hand_contour is None:
#             return None
        
#         # Find the topmost point (lowest y-value)
#         topmost = tuple(hand_contour[hand_contour[:,:,1].argmin()][0])
#         return topmost
    
#     def is_point_in_region(self, point, contour):
#         """Check if point is inside the contour region"""
#         return cv2.pointPolygonTest(contour, point, False) >= 0
    
#     def check_button_press(self, fingertip, button_regions):
#         """Check if fingertip is pressing any button with stability filtering"""
#         current_time = time.time()
        
#         # Cooldown check
#         if current_time - self.last_press_time < self.press_cooldown:
#             return None
        
#         # Check if fingertip is in any button region
#         pressed_button = None
#         if fingertip and button_regions:
#             for color_name, region_info in button_regions.items():
#                 for contour in region_info['contours']:
#                     if self.is_point_in_region(fingertip, contour):
#                         pressed_button = color_name
#                         break
#                 if pressed_button:
#                     break
        
#         # Add to detection buffer for stability
#         self.detection_buffer.append(pressed_button)
#         if len(self.detection_buffer) > self.buffer_size:
#             self.detection_buffer.pop(0)
        
#         # Check for stable detection
#         if len(self.detection_buffer) == self.buffer_size:
#             if all(btn == pressed_button and btn is not None for btn in self.detection_buffer):
#                 # Stable detection - trigger action
#                 self.last_press_time = current_time
#                 self.detection_buffer = []  # Reset buffer
#                 return pressed_button
        
#         return None
    
#     def calibrate_skin_color(self, frame):
#         """Interactive skin color calibration"""
#         print("Skin Calibration Mode:")
#         print("Place your hand in the center of the frame and press SPACE")
#         print("Press ESC to exit calibration")
        
#         h, w = frame.shape[:2]
#         center_x, center_y = w//2, h//2
        
#         # Draw calibration area
#         cv2.rectangle(frame, (center_x-50, center_y-50), (center_x+50, center_y+50), (255, 255, 255), 2)
#         cv2.putText(frame, "Place hand here", (center_x-70, center_y-60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord(' '):  # Spacebar to calibrate
#             # Sample HSV values from center region
#             center_region = frame[center_y-25:center_y+25, center_x-25:center_x+25]
#             hsv_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            
#             # Calculate mean HSV values
#             mean_hsv = np.mean(hsv_region.reshape(-1, 3), axis=0)
            
#             # Set new skin color ranges with some tolerance
#             h_tolerance = 10
#             s_tolerance = 60
#             v_tolerance = 80
            
#             self.skin_lower = np.array([
#                 max(0, mean_hsv[0] - h_tolerance),
#                 max(0, mean_hsv[1] - s_tolerance),
#                 max(0, mean_hsv[2] - v_tolerance)
#             ])
            
#             self.skin_upper = np.array([
#                 min(179, mean_hsv[0] + h_tolerance),
#                 255,
#                 255
#             ])
            
#             print(f"Skin color calibrated! HSV range: {self.skin_lower} to {self.skin_upper}")
#             return True
        
#         return False
    
#     def run(self):
#         """Main loop - THIS IS WHERE THE LIVE VIDEO PROCESSING HAPPENS"""
#         calibration_mode = False
        
#         if not self.cap.isOpened():
#             print("Error: Could not open camera")
#             return
        
#         print("\nStarting Invisible Button System...")
#         print("Make sure you have colored paper squares on your desk:")
#         print("- Red square for 'Next'")
#         print("- Blue square for 'Previous'") 
#         print("- Green square for 'Exit'")
#         print("\nPress any key to continue...")
#         input()
        
#         print("LIVE VIDEO PROCESSING STARTED - Point at colored papers to trigger actions!")
        
#         while True:
#             # LIVE VIDEO CAPTURE: Get current frame from camera RIGHT NOW
#             ret, frame = self.cap.read()
#             if not ret:
#                 print("Error: Could not read frame from camera")
#                 print("Camera might be disconnected or in use by another app")
#                 break
            
#             # REAL-TIME PROCESSING: Analyze this exact moment's video frame
#             # Flip frame horizontally for mirror effect (easier to use)
#             frame = cv2.flip(frame, 1)
            
#             # LIVE INPUT: Check for user keyboard input
#             key = cv2.waitKey(1) & 0xFF
            
#             # Handle key presses
#             if key == ord('q'):
#                 print("Quitting live video processing...")
#                 break
#             elif key == ord('c'):
#                 calibration_mode = not calibration_mode
#                 print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
            
#             if calibration_mode:
#                 # LIVE CALIBRATION: Use current frame for skin color sampling
#                 if self.calibrate_skin_color(frame):
#                     calibration_mode = False
#             else:
#                 # LIVE DETECTION MODE: Process current frame for interactions
                
#                 # STEP 1: Detect colored button regions in current frame
#                 button_regions = self.detect_colored_regions(frame)
                
#                 # STEP 2: Detect hand and fingertip in current frame
#                 hand_contour, hand_mask = self.detect_hand(frame)
#                 fingertip = self.find_fingertip(hand_contour)
                
#                 # STEP 3: Check for button press RIGHT NOW
#                 pressed_button = self.check_button_press(fingertip, button_regions)
#                 if pressed_button:
#                     # IMMEDIATE ACTION: Trigger action as soon as detection happens
#                     button_regions[pressed_button]['action']()
                
#                 # LIVE VISUALIZATION: Draw detection results on current frame
#                 # Draw button regions
#                 for color_name, region_info in button_regions.items():
#                     color_bgr = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0)}
#                     cv2.drawContours(frame, region_info['contours'], -1, color_bgr[color_name], 3)
                    
#                     # Add labels
#                     for contour in region_info['contours']:
#                         M = cv2.moments(contour)
#                         if M["m00"] != 0:
#                             cx = int(M["m10"] / M["m00"])
#                             cy = int(M["m01"] / M["m00"])
#                             cv2.putText(frame, color_name.upper(), (cx-30, cy), 
#                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr[color_name], 2)
                
#                 # Draw fingertip position in real-time
#                 if fingertip:
#                     cv2.circle(frame, fingertip, 8, (255, 255, 0), -1)
#                     cv2.circle(frame, fingertip, 15, (255, 255, 0), 2)
                
#                 # Draw hand contour in real-time
#                 if hand_contour is not None:
#                     cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
                
#                 # Show live hand detection mask in corner
#                 hand_mask_small = cv2.resize(hand_mask, (160, 120))
#                 hand_mask_color = cv2.cvtColor(hand_mask_small, cv2.COLOR_GRAY2BGR)
#                 frame[10:130, 10:170] = hand_mask_color
#                 cv2.putText(frame, "Live Hand Mask", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
#             # Show status on live video
#             status_text = "CALIBRATION MODE" if calibration_mode else "LIVE DETECTION"
#             cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, "Press 'c' to calibrate, 'q' to quit", (10, frame.shape[0]-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
#             # LIVE VIDEO DISPLAY: Show current processed frame
#             cv2.imshow('Invisible Button System - LIVE FEED', frame)
        
#         # Cleanup camera connection
#         print("Closing camera connection...")
#         self.cap.release()
#         cv2.destroyAllWindows()
#         print("System closed")

# # Main execution
# if __name__ == "__main__":
#     system = InvisibleButtonSystem()
#     system.run()

import cv2
import numpy as np
import pyautogui
import time
import math

class GestureControlSystem:
    def __init__(self):
        # CAMERA CONNECTION: Connect to default camera (usually webcam)
        self.cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("ERROR: Cannot access camera!")
            print("Troubleshooting:")
            print("1. Make sure no other apps are using the camera")
            print("2. Try changing camera index (0, 1, 2...)")
            print("3. Check if camera is properly connected")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera connected successfully!")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
        
        # Gesture tracking variables
        self.last_action_time = 0
        self.action_cooldown = 2.0  # 2 seconds between actions
        self.fingertip_history = []
        self.history_size = 15  # Track last 15 positions
        self.gesture_buffer = []
        self.buffer_size = 3  # Need 3 consecutive gesture detections
        
        # Wave detection variables
        self.wave_positions = []
        self.wave_detection_window = 20  # Track positions for wave detection
        self.min_wave_distance = 100  # Minimum horizontal movement for wave
        self.wave_count_threshold = 2  # Number of direction changes needed
        
        # Skin color range (adjust these values for your skin tone)
        self.skin_lower = np.array([0, 20, 70])
        self.skin_upper = np.array([20, 255, 255])
        
        print("Gesture Control System Initialized")
        print("Controls:")
        print("- Point RIGHT: Next slide (Spacebar)")
        print("- Point LEFT: Previous slide (Left arrow)")
        print("- WAVE: Exit presentation (Escape)")
        print("- Press 'q' to quit, 'c' for calibration mode")
    
    def next_slide(self):
        """Action for pointing right"""
        pyautogui.press('space')
        print("🔄 Next slide! (Pointed RIGHT)")
    
    def previous_slide(self):
        """Action for pointing left"""
        pyautogui.press('left')
        print("⬅️ Previous slide! (Pointed LEFT)")
    
    def exit_presentation(self):
        """Action for waving"""
        pyautogui.press('escape')
        print("👋 Exit presentation! (Waved)")
    
    def detect_hand(self, frame):
        """Detect hand using skin color segmentation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find hand contour (largest contour)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assume it's the hand)
            hand_contour = max(contours, key=cv2.contourArea)
            
            # Only consider it a hand if it's large enough
            if cv2.contourArea(hand_contour) > 1000:
                return hand_contour, skin_mask
        
        return None, skin_mask
    
    def find_fingertip_and_direction(self, hand_contour):
        """Find fingertip and determine pointing direction"""
        if hand_contour is None:
            return None, None, None
        
        # Find convex hull and defects to identify fingertip
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(hand_contour, hull)
            
            if defects is not None:
                # Find the point that's furthest from the palm center
                # Get the centroid of the hand
                M = cv2.moments(hand_contour)
                if M["m00"] != 0:
                    palm_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Find the point furthest from palm center
                    max_distance = 0
                    fingertip = None
                    
                    for point in hand_contour:
                        pt = tuple(point[0])
                        distance = math.sqrt((pt[0] - palm_center[0])**2 + (pt[1] - palm_center[1])**2)
                        if distance > max_distance:
                            max_distance = distance
                            fingertip = pt
                    
                    if fingertip and max_distance > 50:  # Minimum distance to be considered pointing
                        # Calculate pointing direction
                        direction_vector = (fingertip[0] - palm_center[0], fingertip[1] - palm_center[1])
                        
                        # Determine if pointing left or right
                        direction = None
                        if abs(direction_vector[0]) > abs(direction_vector[1]):  # More horizontal than vertical
                            if direction_vector[0] > 30:  # Pointing right
                                direction = "RIGHT"
                            elif direction_vector[0] < -30:  # Pointing left
                                direction = "LEFT"
                        
                        return fingertip, palm_center, direction
        
        return None, None, None
    
    def detect_wave_gesture(self, fingertip):
        """Detect waving motion by tracking horizontal movement pattern"""
        if fingertip is None:
            return False
        
        # Add current position to wave tracking
        self.wave_positions.append(fingertip[0])  # Only track x-coordinate
        
        # Keep only recent positions
        if len(self.wave_positions) > self.wave_detection_window:
            self.wave_positions.pop(0)
        
        # Need enough positions to detect wave
        if len(self.wave_positions) < 10:
            return False
        
        # Count direction changes in horizontal movement
        direction_changes = 0
        last_direction = None
        
        for i in range(1, len(self.wave_positions)):
            movement = self.wave_positions[i] - self.wave_positions[i-1]
            
            if abs(movement) > 5:  # Ignore small movements
                current_direction = "right" if movement > 0 else "left"
                
                if last_direction and current_direction != last_direction:
                    direction_changes += 1
                
                last_direction = current_direction
        
        # Calculate total horizontal distance covered
        if len(self.wave_positions) >= 2:
            total_distance = max(self.wave_positions) - min(self.wave_positions)
            
            # Wave detected if we have enough direction changes and distance
            if direction_changes >= self.wave_count_threshold and total_distance > self.min_wave_distance:
                self.wave_positions = []  # Reset after detection
                return True
        
        return False
    
    def process_gesture(self, fingertip, palm_center, pointing_direction):
        """Process detected gesture and trigger actions with stability filtering"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        # Check for wave gesture first (higher priority)
        if self.detect_wave_gesture(fingertip):
            self.last_action_time = current_time
            self.gesture_buffer = []  # Reset buffer
            return "WAVE"
        
        # Process pointing gestures
        detected_gesture = pointing_direction
        
        # Add to gesture buffer for stability
        self.gesture_buffer.append(detected_gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)
        
        # Check for stable detection
        if len(self.gesture_buffer) == self.buffer_size:
            if all(gesture == detected_gesture and gesture is not None for gesture in self.gesture_buffer):
                # Stable detection - trigger action
                self.last_action_time = current_time
                self.gesture_buffer = []  # Reset buffer
                return detected_gesture
        
        return None
    
    def calibrate_skin_color(self, frame):
        """Interactive skin color calibration"""
        print("Skin Calibration Mode:")
        print("Place your hand in the center of the frame and press SPACE")
        print("Press ESC to exit calibration")
        
        h, w = frame.shape[:2]
        center_x, center_y = w//2, h//2
        
        # Draw calibration area
        cv2.rectangle(frame, (center_x-50, center_y-50), (center_x+50, center_y+50), (255, 255, 255), 2)
        cv2.putText(frame, "Place hand here", (center_x-70, center_y-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar to calibrate
            # Sample HSV values from center region
            center_region = frame[center_y-25:center_y+25, center_x-25:center_x+25]
            hsv_region = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            
            # Calculate mean HSV values
            mean_hsv = np.mean(hsv_region.reshape(-1, 3), axis=0)
            
            # Set new skin color ranges with some tolerance
            h_tolerance = 10
            s_tolerance = 60
            v_tolerance = 80
            
            self.skin_lower = np.array([
                max(0, mean_hsv[0] - h_tolerance),
                max(0, mean_hsv[1] - s_tolerance),
                max(0, mean_hsv[2] - v_tolerance)
            ])
            
            self.skin_upper = np.array([
                min(179, mean_hsv[0] + h_tolerance),
                255,
                255
            ])
            
            print(f"Skin color calibrated! HSV range: {self.skin_lower} to {self.skin_upper}")
            return True
        
        return False
    
    def draw_gesture_feedback(self, frame, fingertip, palm_center, pointing_direction):
        """Draw visual feedback for gesture detection"""
        if fingertip and palm_center:
            # Draw fingertip
            cv2.circle(frame, fingertip, 8, (0, 255, 255), -1)
            cv2.circle(frame, fingertip, 15, (0, 255, 255), 2)
            
            # Draw palm center
            cv2.circle(frame, palm_center, 6, (255, 0, 255), -1)
            
            # Draw pointing line
            cv2.line(frame, palm_center, fingertip, (0, 255, 0), 3)
            
            # Show pointing direction
            if pointing_direction:
                direction_color = (0, 255, 0) if pointing_direction in ["LEFT", "RIGHT"] else (0, 0, 255)
                cv2.putText(frame, f"POINTING {pointing_direction}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, direction_color, 2)
        
        # Show wave detection area
        if len(self.wave_positions) > 5:
            wave_status = f"Wave positions: {len(self.wave_positions)}"
            cv2.putText(frame, wave_status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def run(self):
        """Main loop - LIVE GESTURE PROCESSING"""
        calibration_mode = False
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\nStarting Gesture Control System...")
        print("Gesture Instructions:")
        print("- Point your finger clearly LEFT or RIGHT")
        print("- Wave your hand horizontally to exit")
        print("- Keep gestures steady for 1-2 seconds")
        print("\nPress any key to continue...")
        input()
        
        print("LIVE GESTURE DETECTION STARTED!")
        
        while True:
            # LIVE VIDEO CAPTURE
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting gesture control...")
                break
            elif key == ord('c'):
                calibration_mode = not calibration_mode
                print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
            
            if calibration_mode:
                if self.calibrate_skin_color(frame):
                    calibration_mode = False
            else:
                # LIVE GESTURE DETECTION
                
                # Detect hand
                hand_contour, hand_mask = self.detect_hand(frame)
                
                # Find fingertip and pointing direction
                fingertip, palm_center, pointing_direction = self.find_fingertip_and_direction(hand_contour)
                
                # Process gesture and trigger actions
                detected_gesture = self.process_gesture(fingertip, palm_center, pointing_direction)
                
                if detected_gesture:
                    if detected_gesture == "RIGHT":
                        self.next_slide()
                    elif detected_gesture == "LEFT":
                        self.previous_slide()
                    elif detected_gesture == "WAVE":
                        self.exit_presentation()
                
                # LIVE VISUALIZATION
                
                # Draw hand contour
                if hand_contour is not None:
                    cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
                
                # Draw gesture feedback
                self.draw_gesture_feedback(frame, fingertip, palm_center, pointing_direction)
                
                # Show live hand detection mask in corner
                if hand_mask is not None:
                    hand_mask_small = cv2.resize(hand_mask, (160, 120))
                    hand_mask_color = cv2.cvtColor(hand_mask_small, cv2.COLOR_GRAY2BGR)
                    frame[10:130, 10:170] = hand_mask_color
                    cv2.putText(frame, "Live Hand Mask", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show cooldown status
                current_time = time.time()
                cooldown_remaining = max(0, self.action_cooldown - (current_time - self.last_action_time))
                if cooldown_remaining > 0:
                    cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show status
            status_text = "CALIBRATION MODE" if calibration_mode else "GESTURE DETECTION"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to calibrate, 'q' to quit", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display live video
            cv2.imshow('Gesture Control System - LIVE FEED', frame)
        
        # Cleanup
        print("Closing camera connection...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("System closed")

# Main execution
if __name__ == "__main__":
    system = GestureControlSystem()
    system.run()