import cv2
import numpy as np
import pygame
import math
import os
import sys

class HandGestureMusicController:
    def __init__(self, song_path=None):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Initialize Pygame for music
        pygame.mixer.init()
        pygame.init()

        # Music control variables
        self.current_song = song_path
        self.volume = 0.5  # Range 0.0 to 1.0
        self.play_speed = 1.0  # Normal speed
        self.is_playing = False

        # For background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False)

        # For hand tracking
        self.left_hand_roi = None  # Region of interest for left hand
        self.right_hand_roi = None  # Region of interest for right hand
        self.prev_thumb_index_distance = None
        self.prev_hands_distance = None

        # Skin detection thresholds
        self.lower_skin = np.array([0, 25, 80], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Face detection to exclude face regions
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_and_play_song(self):
        """Load and play the selected song"""
        if not self.current_song:
            print("No song specified. Please provide a song path.")
            return False

        try:
            pygame.mixer.music.load(self.current_song)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()
            self.is_playing = True
            print(f"Now playing: {os.path.basename(self.current_song)}")
            return True
        except pygame.error as e:
            print(f"Error loading music: {e}")
            return False

    def detect_hands(self, frame):
        """Detect hands using color thresholding and contour analysis with face exclusion"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for skin color
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Create a face mask to exclude faces
        face_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Exclude face regions from the skin mask
        for (x, y, w, h) in faces:
            # Make the face region black in the face mask
            face_mask[y:y+h, x:x+w] = 0
            # Draw rectangle around faces for visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Apply face mask to skin mask
        skin_mask = cv2.bitwise_and(skin_mask, face_mask)

        # Apply background subtraction to help with segmentation
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        # Combine masks
        combined_mask = cv2.bitwise_and(skin_mask, fg_mask)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and shape features
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Skip too small contours
                continue

            # Calculate convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
            else:
                solidity = 0

            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Hand has these approximate characteristics:
            # - Medium to high solidity (typically 0.7-0.95)
            # - Aspect ratio close to 1 (not too elongated)
            if 0.5 < solidity < 0.95 and 0.5 < aspect_ratio < 2.0:
                valid_contours.append(contour)

        # Sort contours by area (largest first)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]  # At most 2 hands

        # We'll track at most two hands
        hands = []
        for contour in valid_contours:
            # Find convex hull and defects for finger detection
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None and len(defects) >= 1:
                    # Process defects to find fingertips
                    fingertips = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 12000:  # Increase threshold for significant defects
                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])

                            # Filter fingertips based on position
                            # Fingertips are usually at the top of the hand
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])

                                # Add points as fingertips if they're above the center of mass
                                if start[1] < cy:
                                    fingertips.append(start)
                                if end[1] < cy:
                                    fingertips.append(end)

                    # If we have at least 2 fingertips
                    if len(fingertips) >= 2:
                        # Calculate center of hand
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            center = (cx, cy)

                            # Determine if this is left or right hand based on x position
                            is_left = cx < frame.shape[1] // 2

                            hand = {
                                "center": center,
                                "fingertips": fingertips,
                                "is_left": is_left,
                                "contour": contour
                            }

                            # Identify thumb and index finger
                            # Sort fingertips from left to right
                            sorted_fingertips = sorted(fingertips, key=lambda p: p[0])

                            if is_left:
                                # For left hand, thumb is likely the rightmost fingertip
                                # and index finger is likely to its left
                                if len(sorted_fingertips) >= 2:
                                    hand["thumb"] = sorted_fingertips[-1]
                                    hand["index"] = sorted_fingertips[-2]
                            else:
                                # For right hand, thumb is likely the leftmost fingertip
                                # and index finger is likely to its right
                                if len(sorted_fingertips) >= 2:
                                    hand["thumb"] = sorted_fingertips[0]
                                    hand["index"] = sorted_fingertips[1]

                            hands.append(hand)

        return hands, combined_mask

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def process_gestures(self, hands, frame):
        """Process detected hands for gesture control"""
        if not hands:
            return

        left_hand = None
        right_hand = None

        # Identify left and right hands
        for hand in hands:
            if hand["is_left"]:
                left_hand = hand
            else:
                right_hand = hand

        # Volume control with left hand (distance between thumb and index finger)
        if left_hand and "thumb" in left_hand and "index" in left_hand:
            thumb = left_hand["thumb"]
            index = left_hand["index"]

            # Draw the thumb and index finger
            cv2.circle(frame, thumb, 8, (0, 0, 255), -1)
            cv2.circle(frame, index, 8, (0, 255, 0), -1)
            cv2.line(frame, thumb, index, (255, 255, 0), 2)
            cv2.putText(frame, "Thumb", (thumb[0]-10, thumb[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Index", (index[0]-10, index[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Calculate distance between thumb and index finger
            finger_distance = self.calculate_distance(thumb, index)

            # Normalize distance for volume control (0.0 to 1.0)
            # Typical finger distance range might be 10 to 150 pixels
            normalized_distance = np.clip(finger_distance / 150, 0.0, 1.0)

            # Only update if the change is significant
            if self.prev_thumb_index_distance is None or abs(normalized_distance - self.prev_thumb_index_distance) > 0.02:
                self.volume = normalized_distance
                pygame.mixer.music.set_volume(self.volume)
                print(f"Volume set to: {self.volume:.2f}")
                self.prev_thumb_index_distance = normalized_distance

        # Playback speed control with distance between hands
        if left_hand and right_hand:
            left_center = left_hand["center"]
            right_center = right_hand["center"]

            # Draw the hand centers
            cv2.circle(frame, left_center, 10, (255, 0, 0), -1)
            cv2.circle(frame, right_center, 10, (255, 0, 0), -1)
            cv2.line(frame, left_center, right_center, (0, 255, 0), 2)
            cv2.putText(frame, "L", (left_center[0]-5, left_center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "R", (right_center[0]-5, right_center[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            hands_distance = self.calculate_distance(left_center, right_center)

            # Normalize for speed control (0.5x to 1.5x)
            # Typical distance might be 50 to 500 pixels
            speed = 0.5 + (hands_distance / 500)
            speed = np.clip(speed, 0.5, 1.5)

            # Only update if the change is significant
            if self.prev_hands_distance is None or abs(speed - self.play_speed) > 0.05:
                self.play_speed = speed

                # Need to restart playback with new speed
                if self.is_playing and self.current_song:
                    current_pos = pygame.mixer.music.get_pos() / 1000
                    pygame.mixer.music.stop()

                    pygame.mixer.music.load(self.current_song)
                    pygame.mixer.music.set_volume(self.volume)
                    pygame.mixer.music.play(start=current_pos * self.play_speed)

                print(f"Playback speed set to: {self.play_speed:.2f}x")
                self.prev_hands_distance = hands_distance

    def run(self):
        """Main application loop"""
        # First, load and play the song
        if not self.load_and_play_song():
            return

        # Main loop
        print("Calibrating background, please wait...")
        # Allow background subtractor to calibrate
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret:
                self.bg_subtractor.apply(frame)

        print("Calibration complete. Ready!")
        print("Controls:")
        print("- Left hand: Distance between thumb and index finger controls volume")
        print("- Both hands: Distance between hands controls playback speed")
        print("- Press 'q' to quit")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture image from camera.")
                break

            # Flip the image horizontally for a more natural view
            frame = cv2.flip(frame, 1)

            # Process the frame
            hands, mask = self.detect_hands(frame)

            # Process gestures if hands detected
            self.process_gestures(hands, frame)

            # Draw hand contours
            for hand in hands:
                cv2.drawContours(frame, [hand["contour"]], -1, (0, 255, 0), 2)

            # Display info on screen
            cv2.putText(frame, f"Volume: {self.volume:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Speed: {self.play_speed:.2f}x", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.current_song:
                cv2.putText(frame, f"Song: {os.path.basename(self.current_song)}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the mask in a smaller window
            small_mask = cv2.resize(mask, (frame.shape[1]//3, frame.shape[0]//3))
            frame[10:10+small_mask.shape[0], frame.shape[1]-10-small_mask.shape[1]:frame.shape[1]-10] = \
                cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)

            # Display the image
            cv2.imshow('Hand Gesture Music Control', frame)

            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def print_usage():
    print("Usage: python hand_gesture_controller.py <path_to_music_file>")
    print("Example: python hand_gesture_controller.py music/song.mp3")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    song_path = sys.argv[1]
    if not os.path.exists(song_path):
        print(f"Error: The file '{song_path}' does not exist.")
        sys.exit(1)

    controller = HandGestureMusicController(song_path)
    controller.run()
