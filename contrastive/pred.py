import argparse
from pathlib import Path
from model.Model import Model
from typing import Union, List

def predict(model: Model, image_path: Union[str, Path, List[Union[str, Path]]]) -> None:
    image_path = Path(image_path) if isinstance(image_path, str) else image_path
    
    # Handle single file
    if image_path.is_file():
        result = model(str(image_path))
        print(f"Prediction result: {result}")
    elif image_path.is_dir():
        image_files = list(image_path.glob('*.[jJ][pP][gG]')) + \
                     list(image_path.glob('*.[jJ][pP][eE][gG]')) + \
                     list(image_path.glob('*.[pP][nN][gG]'))
            
        print(f"Found {len(image_files)} images to process")
        
        for i, img_file in enumerate(image_files, 1):
            result = model(str(img_file))
            print(f"{i}/{len(image_files)} {img_file.name}: {result}")

def main():
    parser = argparse.ArgumentParser(
        description="Image classification using different model backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to model file (.pt, .ckpt, .onnx, .engine)"
    )
    
    parser.add_argument(
        "model_backend",
        choices=["torch", "ort", "trt"],
        help="Inference backend: torch=PyTorch, ort=ONNX Runtime, trt=TensorRT"
    )
    
    parser.add_argument(
        "embedding_dict",
        type=Path,
        help="Path to precomputed embedding dictionary"
    )
    
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to image file or directory with images"
    )
    
    args = parser.parse_args()

    # Initialize model
    model = Model(
        model_path=str(args.model_path),
        format=args.model_backend,
        dict_path=str(args.embedding_dict),
    )

    # Run prediction
    predict(model, args.image_path)

if __name__ == "__main__":
    main()