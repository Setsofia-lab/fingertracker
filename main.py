import cv2
import numpy as np
import pyautogui
import time

class InvisibleButtonSystem:
    def __init__(self):
        # CAMERA CONNECTION: Connect to default camera (usually webcam)
        # 0 = default camera, 1 = second camera, etc.
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
        
        self.last_press_time = 0
        self.press_cooldown = 1.0  # 1 second between presses
        self.detection_buffer = []
        self.buffer_size = 3  # Need 3 consecutive detections
        
        # Color ranges for different buttons (HSV format)
        self.colors = {
            'red': {
                'lower': np.array([0, 120, 70]),
                'upper': np.array([10, 255, 255]),
                'action': self.next_slide
            },
            'blue': {
                'lower': np.array([100, 150, 50]),
                'upper': np.array([140, 255, 255]),
                'action': self.previous_slide
            },
            'green': {
                'lower': np.array([40, 40, 40]),
                'upper': np.array([80, 255, 255]),
                'action': self.exit_presentation
            }
        }
        
        # Skin color range (adjust these values for your skin tone)
        self.skin_lower = np.array([0, 20, 70])
        self.skin_upper = np.array([20, 255, 255])
        
        print("Invisible Button System Initialized")
        print("Controls:")
        print("- Red button: Next slide (Spacebar)")
        print("- Blue button: Previous slide (Left arrow)")
        print("- Green button: Exit presentation (Escape)")
        print("- Press 'q' to quit, 'c' for calibration mode")
    
    def next_slide(self):
        """Action for red button"""
        pyautogui.press('space')
        print("Next slide!")
    
    def previous_slide(self):
        """Action for blue button"""
        pyautogui.press('left')
        print("Previous slide!")
    
    def exit_presentation(self):
        """Action for green button"""
        pyautogui.press('escape')
        print("Exit presentation!")
    
    def detect_colored_regions(self, frame):
        """Detect all colored button regions"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        button_regions = {}
        
        for color_name, color_info in self.colors.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            
            # Clean up noise
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter by area (remove small noise)
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    valid_contours.append(contour)
            
            if valid_contours:
                button_regions[color_name] = {
                    'contours': valid_contours,
                    'action': color_info['action']
                }
        
        return button_regions
    
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
    
    def find_fingertip(self, hand_contour):
        """Find fingertip as the topmost point of hand contour"""
        if hand_contour is None:
            return None
        
        # Find the topmost point (lowest y-value)
        topmost = tuple(hand_contour[hand_contour[:,:,1].argmin()][0])
        return topmost
    
    def is_point_in_region(self, point, contour):
        """Check if point is inside the contour region"""
        return cv2.pointPolygonTest(contour, point, False) >= 0
    
    def check_button_press(self, fingertip, button_regions):
        """Check if fingertip is pressing any button with stability filtering"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_press_time < self.press_cooldown:
            return None
        
        # Check if fingertip is in any button region
        pressed_button = None
        if fingertip and button_regions:
            for color_name, region_info in button_regions.items():
                for contour in region_info['contours']:
                    if self.is_point_in_region(fingertip, contour):
                        pressed_button = color_name
                        break
                if pressed_button:
                    break
        
        # Add to detection buffer for stability
        self.detection_buffer.append(pressed_button)
        if len(self.detection_buffer) > self.buffer_size:
            self.detection_buffer.pop(0)
        
        # Check for stable detection
        if len(self.detection_buffer) == self.buffer_size:
            if all(btn == pressed_button and btn is not None for btn in self.detection_buffer):
                # Stable detection - trigger action
                self.last_press_time = current_time
                self.detection_buffer = []  # Reset buffer
                return pressed_button
        
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
    
    def run(self):
        """Main loop - THIS IS WHERE THE LIVE VIDEO PROCESSING HAPPENS"""
        calibration_mode = False
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\nStarting Invisible Button System...")
        print("Make sure you have colored paper squares on your desk:")
        print("- Red square for 'Next'")
        print("- Blue square for 'Previous'") 
        print("- Green square for 'Exit'")
        print("\nPress any key to continue...")
        input()
        
        print("LIVE VIDEO PROCESSING STARTED - Point at colored papers to trigger actions!")
        
        while True:
            # LIVE VIDEO CAPTURE: Get current frame from camera RIGHT NOW
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                print("Camera might be disconnected or in use by another app")
                break
            
            # REAL-TIME PROCESSING: Analyze this exact moment's video frame
            # Flip frame horizontally for mirror effect (easier to use)
            frame = cv2.flip(frame, 1)
            
            # LIVE INPUT: Check for user keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                print("Quitting live video processing...")
                break
            elif key == ord('c'):
                calibration_mode = not calibration_mode
                print(f"Calibration mode: {'ON' if calibration_mode else 'OFF'}")
            
            if calibration_mode:
                # LIVE CALIBRATION: Use current frame for skin color sampling
                if self.calibrate_skin_color(frame):
                    calibration_mode = False
            else:
                # LIVE DETECTION MODE: Process current frame for interactions
                
                # STEP 1: Detect colored button regions in current frame
                button_regions = self.detect_colored_regions(frame)
                
                # STEP 2: Detect hand and fingertip in current frame
                hand_contour, hand_mask = self.detect_hand(frame)
                fingertip = self.find_fingertip(hand_contour)
                
                # STEP 3: Check for button press RIGHT NOW
                pressed_button = self.check_button_press(fingertip, button_regions)
                if pressed_button:
                    # IMMEDIATE ACTION: Trigger action as soon as detection happens
                    button_regions[pressed_button]['action']()
                
                # LIVE VISUALIZATION: Draw detection results on current frame
                # Draw button regions
                for color_name, region_info in button_regions.items():
                    color_bgr = {'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0)}
                    cv2.drawContours(frame, region_info['contours'], -1, color_bgr[color_name], 3)
                    
                    # Add labels
                    for contour in region_info['contours']:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(frame, color_name.upper(), (cx-30, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr[color_name], 2)
                
                # Draw fingertip position in real-time
                if fingertip:
                    cv2.circle(frame, fingertip, 8, (255, 255, 0), -1)
                    cv2.circle(frame, fingertip, 15, (255, 255, 0), 2)
                
                # Draw hand contour in real-time
                if hand_contour is not None:
                    cv2.drawContours(frame, [hand_contour], -1, (0, 255, 255), 2)
                
                # Show live hand detection mask in corner
                hand_mask_small = cv2.resize(hand_mask, (160, 120))
                hand_mask_color = cv2.cvtColor(hand_mask_small, cv2.COLOR_GRAY2BGR)
                frame[10:130, 10:170] = hand_mask_color
                cv2.putText(frame, "Live Hand Mask", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show status on live video
            status_text = "CALIBRATION MODE" if calibration_mode else "LIVE DETECTION"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to calibrate, 'q' to quit", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # LIVE VIDEO DISPLAY: Show current processed frame
            cv2.imshow('Invisible Button System - LIVE FEED', frame)
        
        # Cleanup camera connection
        print("Closing camera connection...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("System closed")

# Main execution
if __name__ == "__main__":
    system = InvisibleButtonSystem()
    system.run()
