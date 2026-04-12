import cv2
from src.feature_engineering.build_features import FeatureBuilder

def main():
    # Correct path (your image is inside src/)
    image_path = "src/test.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("❌ Image not found!")

    # Initialize Feature Builder
    builder = FeatureBuilder()

    # Extract features
    features = builder.extract_features(image)

    print("✅ Feature extraction successful!")
    print("Feature vector shape:", features.shape)


if __name__ == "__main__":
    main()