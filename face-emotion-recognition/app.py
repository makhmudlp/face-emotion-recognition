import torch
import gradio as gr
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from models.resnet import EmotionResNet

# ============================================================
# CONFIG
# ============================================================
DEVICE     = torch.device("cpu")
MODEL_PATH = "models/best_resnet.pth"
EMOTIONS   = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJIS     = ["😠", "🤢", "😨", "😊", "😐", "😢", "😮"]

EMOTION_COLORS = {
    "Angry":    (0, 0, 255),
    "Disgust":  (0, 140, 255),
    "Fear":     (128, 0, 128),
    "Happy":    (0, 255, 0),
    "Neutral":  (255, 255, 255),
    "Sad":      (255, 0, 0),
    "Surprise": (0, 255, 255),
}

# ============================================================
# LOAD MODEL
# ============================================================
model = EmotionResNet(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded!")

# ============================================================
# TRANSFORM
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================
# FACE DETECTOR
# ============================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================================
# PREDICT
# ============================================================
def predict(image):
    if image is None:
        return None, "Please upload an image", {}

    # PIL → numpy RGB
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    confidences = {f"{EMOJIS[i]} {e}": 0.0 for i, e in enumerate(EMOTIONS)}

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    if len(faces) == 0:
        cv2.putText(frame_bgr, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        result = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result), "No face detected", {}

    # Use largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face_crop = frame_bgr[y:y+h, x:x+w]
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    pil_img   = Image.fromarray(face_gray)
    tensor    = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top_idx    = probs.argmax()
    emotion    = EMOTIONS[top_idx]
    emoji      = EMOJIS[top_idx]
    confidence = probs[top_idx]
    color      = EMOTION_COLORS[emotion]

    confidences = {
        f"{EMOJIS[i]} {e}": float(probs[i])
        for i, e in enumerate(EMOTIONS)
    }

    # Draw on image
    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 3)
    cv2.rectangle(frame_bgr, (x, y-45), (x+w, y), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        f"{emotion} {confidence*100:.0f}%",
        (x+5, y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
    )

    result = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    label  = f"{emoji} {emotion} — {confidence*100:.0f}% confident"

    return Image.fromarray(result), label, confidences


# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks(title="Face Emotion Recognition") as demo:

    gr.Markdown("""
    # 😊 Face Emotion Recognition
    **Model:** Fine-tuned ResNet18 on FER2013 &nbsp;|&nbsp; **Accuracy:** 68.35%
    
    Upload a photo or take one with your camera to detect the emotion!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                sources=["webcam", "upload"],  # upload or take photo
                type="pil",
                label="📷 Upload or Take Photo",
                interactive=True
            )
            submit_btn = gr.Button("🔍 Detect Emotion", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="👤 Result")
            emotion_text = gr.Textbox(
                label="🎯 Detected Emotion",
                interactive=False,
                text_align="center"
            )
            confidence_bars = gr.Label(
                label="📊 Confidence",
                num_top_classes=7
            )

    # Example images
    gr.Examples(
        examples=[],
        inputs=input_image
    )

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[output_image, emotion_text, confidence_bars]
    )

    gr.Markdown("""
    ---
    **Tips:** 💡 Good lighting &nbsp;|&nbsp; 👤 Clear face visible &nbsp;|&nbsp; 😁 Try different expressions!
    """)

demo.launch()