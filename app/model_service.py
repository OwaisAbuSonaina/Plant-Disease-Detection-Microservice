import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json


class PlantDiseaseInference:
    def __init__(self, model_path: str, class_names_path: str):
        # 1. Define Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Recreate Model Architecture (Must match training!)
        self.model = models.convnext_tiny(weights=None)

        # 3. Reconfigure the Classifier Head
        num_classes = 38
        self.model.classifier[2] = nn.Linear(
            self.model.classifier[2].in_features, num_classes
        )

        # 4. Load State Dict
        # map_location ensures it loads on CPU if GPU is missing
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (freezes Batch Norm/Dropout)

        # 5. Load Class Names
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)

        # 6. Define Preprocessing Transforms (Same as validation set)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, image_bytes):
        # Open Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        tensor = self.transform(image).unsqueeze(
            0
        )  # Add batch dimension -> [1, 3, 224, 224]
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Post-process results
        confidence, predicted_class_idx = torch.max(probabilities, 1)
        predicted_class = self.class_names[predicted_class_idx.item()]

        return {"class": predicted_class, "confidence": float(confidence.item())}
