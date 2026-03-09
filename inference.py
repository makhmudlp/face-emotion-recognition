
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import mediapipe as mp
from models.resnet import EmotionResNet

DEVICE='mps' if torch.mps.is_available() else 'cpu'
MODEL_PATH='models/best_resnet.pth'

model=EmotionResNet(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_emotion(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    input_to_model = transform(gray).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_to_model)
        probs = torch.softmax(output, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][idx].item()

    return EMOTIONS[idx], confidence


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image=cv2.flip(image, 1)

        mesh_results = face_mesh.process(image)
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        # Step 1: prepare frame for MediaPipe
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Step 2: draw and predict
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                # Get bounding box in pixel coords
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Crop face and predict
                face_crop = image[y:y+bh, x:x+bw]
                if face_crop.size > 0:
                    emotion, confidence = predict_emotion(face_crop)
                    # Draw box and label
                    cv2.putText(image, f"Emotion: {emotion}",
                               (20,50), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image,f"Confidence: {confidence*100:.0f}%",
                               (20,90), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection',image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
