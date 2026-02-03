import os
from datetime import datetime
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from python.src.ModelTraining import train_model, get_predictions


def run_pipeline():
    print("--- 1. Starting Model Training ---")
    model = train_model()

    print("\n--- 2. Evaluating Model Performance ---")
    get_predictions(model)

    print("\n--- 3. Exporting Versioned ONNX Model ---")
    model_dir = "../model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"photo_quality_model_{timestamp}.onnx"
    file_path = os.path.join(model_dir, filename)

    initial_type = [('float_input', FloatTensorType([None, 3]))]

    onx = to_onnx(model, initial_types=initial_type)

    with open(file_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"SUCCESS: Exported {filename} to {model_dir}")


if __name__ == "__main__":
    run_pipeline()