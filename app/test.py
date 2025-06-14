# model/test.py
from model import Preprocessor, OnnxModel

def test_prediction(image_path):
    pre = Preprocessor()
    model = OnnxModel()
    img_tensor = pre.preprocess(image_path)
    prediction = model.predict(img_tensor)
    print(f"Predicted Class ID: {prediction}")

if __name__ == "__main__":
    test_prediction("C:\\Users\\Kirti Bhatt\\Desktop\\cerebrium-model-deployment\\images\\n01440764_tench.jpeg")
    test_prediction("C:\\Users\\Kirti Bhatt\\Desktop\\cerebrium-model-deployment\\images\\n01667114_mud_turtle.JPEG")
