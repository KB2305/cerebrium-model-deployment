
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from app.pytorch_model import Classifier, BasicBlock

def convert():
    # 1. Load your model structure
    model = Classifier(BasicBlock, [2, 2, 2, 2])

    # 2. Load the trained weights (learned brain)
    model.load_state_dict(torch.load("./weights/pytorch_model_weights.pth"))

    # 3. Tell the model to be in "inference" mode
    model.eval()

    # 4. Create a fake input that looks like an image (for ONNX export)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 5. Export model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        "./weights/model.onnx",  # This will be the new file
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

    print("Model saved as ONNX at ./weights/model.onnx")

if __name__ == "__main__":
    convert()
