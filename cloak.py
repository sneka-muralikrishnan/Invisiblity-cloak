import numpy as np
import cv2
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
time.sleep(3)  # Give the camera time to adjust

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

background = None

# Capturing the background
for i in range(60):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture background frame.")
        continue
    background = frame

if background is not None and background.size > 0:
    background = np.flip(background, axis=1)
else:
    print("Error: Background not captured properly.")
    cap.release()
    exit()

# Main loop for invisibility effect
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color range for all shades of blue
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological transformations to remove noise
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    # Create inverse mask
    mask2 = cv2.bitwise_not(mask1)

    # Segment out the blue color and replace it with the background
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the result
    cv2.imshow('Invisible Cloak', final_output)

    # Exit on 'Esc' key press
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
