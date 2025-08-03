from pathlib import Path
from model.Model import Model
import argparse

def build_data_dict(model: Model, data_path: Path, dest_path: Path):
    data_path = data_path.glob('**/*')
    samples_images = []
    samples_labels = []
    class_to_idx = {}
    classes = [x for x in data_path if x.is_dir()]
    for idx, class_name in enumerate(classes):
        class_to_idx[class_name] = idx

    idx_to_class = {v: k.name for k, v in class_to_idx.items()}

    for class_name in class_to_idx:
        class_dir = Path(class_name).glob('*.jpg')

        for file_name in class_dir:
            samples_images.append(file_name)
            samples_labels.append(class_to_idx[class_name])

    model.build_dict(samples_images, samples_labels, dest_path, idx_to_class)

def main():
    parser = argparse.ArgumentParser(
            description="Build embedding dictionary from image data using a trained model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to model file (supports .pt, .ckpt, .onnx, .engine)"
    )
    parser.add_argument(
        "model_backend",
        choices=["torch", "ort", "trt"],
        help="Backend to use: torch (PyTorch), ort (ONNX Runtime), trt (TensorRT)"
    )
    parser.add_argument(
        "output_dict_path",
        type=Path,
        help="Path to save the precomputed embedding dictionary"
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to directory containing image data organized by classes"
    )
    args = parser.parse_args()
    model = Model(args.model_path, args.model_backend)

    build_data_dict(model, args.data_path, args.output_dict_path)

if __name__ == "__main__":
    main()