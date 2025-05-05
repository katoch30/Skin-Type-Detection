import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# --- Load your trained CNN model ---

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Dummy input to infer output size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)  # adjust if your input size is different
            out = self.conv3(self.conv2(self.conv1(dummy)))
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(32768, 128)  # match your trained model
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = ImprovedCNN(num_classes=3)
model.load_state_dict(torch.load("best_model_fold_5.pth", map_location=torch.device('cpu')))
model.eval()

# --- Define Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

labels_map = ['dry', 'normal', 'oily']

# --- Helper to predict region ---
def predict_region(cropped_img):
    img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return labels_map[predicted.item()]

# --- Start webcam ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 's' to take a snapshot and predict. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Split into regions: top (T-zone), bottom left, bottom right
        tzone = frame[y:y + h//3, x:x + w]               # Top part
        left_cheek = frame[y + h//3:y + h, x:x + w//2]   # Bottom left
        right_cheek = frame[y + h//3:y + h, x + w//2:x + w]  # Bottom right

    cv2.imshow("Skin Type Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected.")
        else:
            print("Detecting skin type...")

            tzone_label = predict_region(tzone)
            left_label = predict_region(left_cheek)
            right_label = predict_region(right_cheek)

            # Determine final result
            all_labels = [tzone_label, left_label, right_label]
            print("T-zone:", tzone_label)
            print("Left cheek:", left_label)
            print("Right cheek:", right_label)

            if len(set(all_labels)) > 1 and 'oily' in all_labels and 'dry' in all_labels:
                print("Final Prediction: Combination Skin")
            else:
                # Majority vote
                final_label = max(set(all_labels), key=all_labels.count)
                print(f"Final Prediction: {final_label.capitalize()} Skin")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
