import time
import argparse
from pathlib import Path
from typing import List
from model.Model import Model

def evaluate_model(
    model: Model, 
    data_path: Path, 
    countries: List[str],
) -> None:
    correct = 0
    total_time = 0.0
    
    print("\nStarting evaluation...")
    print(f"Countries to test: {len(countries)}")
    print("=" * 50)
    
    for country in countries:
        img_path = data_path / country / f"{country}_90.jpg"
            
        start_time = time.time()
        prediction = model(str(img_path))
        elapsed = time.time() - start_time
        
        # Check if prediction is correct
        is_correct = country == prediction
        correct += int(is_correct)
        total_time += elapsed
        
        print(f"Country: {country:<5} | Predicted: {prediction:<5} | "
                f"Correct: {'✓' if is_correct else '✗'} | "
                f"Time: {elapsed:.4f}s")
                  
    
    # Calculate final metrics
    accuracy = correct / len(countries) if countries else 0.0
    avg_time = total_time / len(countries) if countries else 0.0
    
    print("=" * 50)
    print(f"\nEvaluation Results:")
    print(f"- Countries tested: {len(countries)}")
    print(f"- Correct predictions: {correct}")
    print(f"- Accuracy: {accuracy:.4f} ({correct}/{len(countries)})")
    print(f"- Average prediction time: {avg_time:.4f}s")
    print(f"- Total evaluation time: {total_time:.2f}s")


def main():
    # List of supported country codes
    COUNTRIES = [
        "UZB", "BEL", "BGR", "BLR", "CAN", "CHL", "DOM", "ESP", 
        "EST", "GBR", "HUN", "IDN", "IRL", "ITA", "KAZ", "KGZ", 
        "MDA", "MEX", "NLD", "POL", "SVK", "SWE", "USA"
    ]
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate country recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "model_path",
        type=Path,
        default="tulip-B-15-224.ckpt.onnx",
        help="Path to model file (.pt, .ckpt, .onnx, .engine)",
    )
    
    parser.add_argument(
        "model_backend",
        choices=["torch", "ort", "trt"],
        default="ort",
        help="Inference backend (torch=PyTorch, ort=ONNX Runtime, trt=TensorRT)"
    )
    
    parser.add_argument(
        "embedding_dict",
        type=Path,
        default="data_dict.pt",
        help="Path to precomputed embedding dictionary in pt format"
    )
    
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to directory containing country folders with test images"
    )
    
    # Optional arguments
    parser.add_argument(
        "--countries",
        nargs="+",
        default=COUNTRIES,
        help="List of country codes to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    model = Model(
        model_path=str(args.model_path),
        format=args.model_backend,
        dict_path=str(args.embedding_dict)
    )
    
    # Run evaluation
    evaluate_model(
        model=model,
        data_path=args.data_path,
        countries=args.countries,
    )

if __name__ == "__main__":
    main()
