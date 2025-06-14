import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "model.onnx")

class Preprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).numpy()
        return tensor

# class OnnxModel:
#     def __init__(self, model_path = "C:\\Users\\Kirti Bhatt\\Desktop\\cerebrium-model-deployment\\weights\\model.onnx"):
#         self.session = ort.InferenceSession(model_path)
#         self.input_name = self.session.get_inputs()[0].name
# import os

class OnnxModel:
    def __init__(self, model_path=None):
        if model_path is None:
            # Assume model.onnx is in /app/weights/ when container runs
            model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "model.onnx")
            model_path = os.path.abspath(model_path)
        
        self.session = ort.InferenceSession(model_path)

    def predict(self, input_tensor):
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predicted_class = np.argmax(outputs[0])
        return predicted_class
