import cv2
from src.feature_engineering.build_features import FeatureBuilder

# Load image
image = cv2.imread("test.jpg")  # put any image here

if image is None:
    raise ValueError("Image not found!")

# Initialize
builder = FeatureBuilder()

# Extract features
features = builder.extract_features(image)

print("Feature vector shape:", features.shape)