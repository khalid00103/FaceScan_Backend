import cv2
import dlib
import numpy as np
import os

# Load the pre-trained face detector and shape predictor models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to detect faces and facial landmarks
def detect_faces_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    landmarks_list = []
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks_list.append([(p.x, p.y) for p in landmarks.parts()])
    return faces, landmarks_list

# Function to calculate facial aspect ratios and determine face shape
def calculate_face_shape(landmarks):
    # Key landmarks
    jawline_points = np.array(landmarks[4:13])
    forehead_width = np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[26]))
    jawline_width = np.linalg.norm(jawline_points[0] - jawline_points[-1])
    cheekbones_width = np.linalg.norm(np.array(landmarks[2]) - np.array(landmarks[14]))
    face_height = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[25]))

    # Standardize the face dimensions
    standardized_height = 100.0
    scale_factor = standardized_height / face_height

    standardized_forehead_width = forehead_width * scale_factor
    standardized_jawline_width = jawline_width * scale_factor
    standardized_cheekbones_width = cheekbones_width * scale_factor

    # Determine face shape using standardized measurements
    if standardized_cheekbones_width > standardized_forehead_width + (20 * scale_factor) and standardized_forehead_width > standardized_jawline_width + (15 * scale_factor):
        return "Heart"
    elif abs(standardized_forehead_width - standardized_cheekbones_width) <= (30 * scale_factor) and abs(standardized_forehead_width - standardized_jawline_width) <= (30 * scale_factor) and abs(standardized_cheekbones_width - standardized_jawline_width) <= (30 * scale_factor) and standardized_height > standardized_cheekbones_width + (17 * scale_factor):
        return "Oblong"
    elif abs(standardized_forehead_width - standardized_jawline_width) <= (30 * scale_factor) and abs(standardized_cheekbones_width - standardized_jawline_width) <= (37 * scale_factor):
        return "Square"
    elif standardized_cheekbones_width - max(standardized_forehead_width, standardized_jawline_width) > (25 * scale_factor) and standardized_height > standardized_cheekbones_width + (20 * scale_factor):
        return "Oval"
    elif abs(standardized_forehead_width - standardized_jawline_width) <= (30 * scale_factor) and standardized_cheekbones_width - max(standardized_forehead_width, standardized_jawline_width) > (20 * scale_factor):
        return "Round"
    else:
        return "Unknown"

# Specify input and output folder paths
input_folder = "Input Images"

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        faces, landmarks_list = detect_faces_landmarks(image)
        for landmarks in landmarks_list:
            face_shape = calculate_face_shape(landmarks)
            print(face_shape)
            # Do something with the face_shape if needed

print("Processing complete.")
