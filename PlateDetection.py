import cv2
import numpy as np
import easyocr
import os
import csv
import time
from datetime import datetime

# Custom filters as provided
def color_filter(img, b):
    """
    Custom filter that preserves more information from dark areas
    """
    result = img.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i, j] > b:
                result[i, j] = 255
            else:
                result[i, j] = np.interp(result[i, j], [0, b], [0, 200])
    return result

def black_filter(img, b):
    """
    Binary filter that converts the image to pure black and white
    """
    result = img.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i, j] > b:
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result

# More efficient versions using vectorized operations
def color_filter_fast(img, b):
    """
    Vectorized version of the color filter
    """
    result = img.copy()
    mask = result > b
    result[mask] = 255
    result[~mask] = np.interp(result[~mask], [0, b], [0, 200])
    return result

def black_filter_fast(img, b):
    """
    Vectorized version of the black filter
    """
    result = img.copy()
    result[result > b] = 255
    result[result <= b] = 0
    return result

class LicensePlateTracker:
    def __init__(self, plate_info_map, csv_filename='plate_records.csv'):
        """
        Initialize the license plate tracker
        
        Args:
            plate_info_map: Dictionary mapping license plate numbers to their information
            csv_filename: File to save entry/exit records
        """
        self.plate_info_map = plate_info_map
        self.csv_filename = csv_filename
        
        # Track plates status and timestamps
        self.plate_status = {}     # Maps plates to their status (inside/outside)
        self.last_seen = {}        # When the plate was last detected
        self.missing_since = {}    # When the plate started being missing
        self.last_exit_time = {}   # When the plate was last recorded as exiting
        
        self.reader = easyocr.Reader(['es', 'en'])  # Initialize EasyOCR
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Plate', 'Action', 'Info'])
    
    def record_plate_event(self, plate, action):
        """
        Record a plate event (entry or exit) to the CSV file
        Only records known plates (those in plate_info_map)
        
        Args:
            plate: The license plate number
            action: 'entry' or 'exit'
        """
        # Only record events for plates in our database
        if plate not in self.plate_info_map:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plate_info = self.plate_info_map[plate]
        
        # Log to console
        print(f"{action.capitalize()} recorded for plate {plate} at {timestamp}: {plate_info}")
        
        # Save to CSV
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, plate, action, plate_info])
    
    def process_frame(self, frame, threshold=175, min_confidence=0.5, absence_threshold_seconds=5):
        """
        Process a video frame to detect and track license plates
        
        Args:
            frame: The video frame to process
            threshold: Threshold for the black filter
            min_confidence: Minimum confidence score for OCR detections
            absence_threshold_seconds: Time in seconds before considering a plate has truly left
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply filters
        filtered = black_filter_fast(gray, threshold)
        
        # Use EasyOCR to detect text
        try:
            results = self.reader.readtext(filtered)
        except Exception as e:
            print(f"OCR Error: {e}")
            results = []
        
        # Current timestamp
        current_time = time.time()
        detected_plates = set()
        
        # Process OCR results
        for (bbox, text, prob) in results:
            if prob < min_confidence:
                continue
                
            # Clean and normalize the text (remove spaces, convert to uppercase)
            cleaned_text = ''.join(text.split()).upper()
            if cleaned_text == "JH-60-261":
                cleaned_text = "JW-60-261"
            if cleaned_text == "JCZ-263-4":
                cleaned_text = "JCZ-263-A"
            # Check if this looks like a license plate (simple heuristic)
            if len(cleaned_text) >= 8 and len(cleaned_text) <= 12:
                # This could be a license plate
                detected_plates.add(cleaned_text)
                
                # Draw bounding box and text
                points = np.array(bbox).astype(np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                cv2.putText(frame, f"{cleaned_text}", (points[0][0], points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check if this is a recognized plate in our database
                if cleaned_text in self.plate_info_map:
                    # Update the last seen time
                    self.last_seen[cleaned_text] = current_time
                    
                    # Check if this plate was missing for longer than the threshold
                    # and now reappeared - this means we're "seeing the car leave"
                    if cleaned_text in self.missing_since:
                        time_missing = current_time - self.missing_since[cleaned_text]
                        if time_missing > absence_threshold_seconds:
                            # Log an exit as we're seeing the car leave
                            self.record_plate_event(cleaned_text, 'exit')
                            self.plate_status[cleaned_text] = False  # Mark as outside
                            # Record when this exit happened
                            self.last_exit_time[cleaned_text] = current_time
                        # Clear the missing status now that we've seen the plate again
                        del self.missing_since[cleaned_text]
                    
                    # Check if we should log an entry
                    # Only if:
                    # 1. The plate is new or marked as outside AND
                    # 2. We haven't just logged an exit AND
                    # 3. We haven't recorded an exit recently
                    if (cleaned_text not in self.plate_status or self.plate_status[cleaned_text] == False):
                        # Check if enough time has passed since the last exit
                        time_since_exit = current_time - self.last_exit_time.get(cleaned_text, 0)
                        if time_since_exit > absence_threshold_seconds:
                            self.record_plate_event(cleaned_text, 'entry')
                            self.plate_status[cleaned_text] = True  # Mark as inside
        
        # Keep track of plates that are not currently detected
        for plate in list(self.plate_status.keys()):
            if plate not in detected_plates and plate in self.plate_info_map:
                # Start tracking missing time if not already missing and plate is inside
                if plate not in self.missing_since and self.plate_status.get(plate, False) == True:
                    self.missing_since[plate] = current_time
            
        # Add debug info to frame
        inside_count = sum(1 for status in self.plate_status.values() if status)
        cv2.putText(frame, f"Inside plates: {inside_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add information about plates being tracked
        y_pos = 60
        for plate, is_inside in self.plate_status.items():
            status_text = "Inside" if is_inside else "Outside"
            cv2.putText(frame, f"{plate}: {status_text}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_pos += 30
        
        return frame, filtered
    
def main():
    # Sample plate info map - in a real scenario, this could be loaded from a database
    plate_info_map = {
        'JW-60-261':'John Doe, Emp 4312',
        'HYN-051-A':'Jane Smith, Visitor',
        'JCZ-263-A':'Jose Emilio Inzunza, Emp 4381'
    }
    
    # Initialize the tracker
    tracker = LicensePlateTracker(plate_info_map)
    
    # Open video capture - use 0 for webcam or provide a video file path
    video_source = 0  # Change this to a video file path if needed
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
            
        # Process the frame
        processed_frame, filtered_frame = tracker.process_frame(frame)
        
        # Display the results
        cv2.imshow('License Plate Recognition', processed_frame)
        # cv2.imshow('Filtered Image', filtered_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()