import argparse

from Model import Model

def main():
    parser = argparse.ArgumentParser(
        description="OCR-based Passport Country Recognition System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to passport image file"
    )
    
    parser.add_argument(
        "--keywords",
        type=str,
        default="country_keywords.json",
        help="Path to JSON file with country keywords"
    )
    args = parser.parse_args()

    model = Model(args.keywords)
    result = model(args.image_path)
    print(f"Country code: {result}")

if __name__ == "__main__":
    main()