import numpy as np

# Import BOTH class and function styles
from ..utils import lbp_features, glcm_features, fft_features


class FeatureBuilder:
    def __init__(self, use_lbp=True, use_glcm=True, use_fft=True):

        self.extractors = []

        if use_lbp:
            # Try class first, else fallback to function
            if hasattr(lbp_features, "LBPExtractor"):
                self.extractors.append(lbp_features.LBPExtractor())
            elif hasattr(lbp_features, "extract_lbp_features"):
                self.extractors.append(lbp_features.extract_lbp_features)

        if use_glcm:
            if hasattr(glcm_features, "GLCMExtractor"):
                self.extractors.append(glcm_features.GLCMExtractor())
            elif hasattr(glcm_features, "extract_glcm_features"):
                self.extractors.append(glcm_features.extract_glcm_features)

        if use_fft:
            if hasattr(fft_features, "FFTExtractor"):
                self.extractors.append(fft_features.FFTExtractor())
            elif hasattr(fft_features, "extract_fft_features"):
                self.extractors.append(fft_features.extract_fft_features)

        if len(self.extractors) == 0:
            raise ValueError("No feature extractors selected!")

    def extract_features(self, image):
        feature_list = []

        for extractor in self.extractors:
            try:
                # Handle BOTH types
                if callable(extractor):
                    features = extractor(image)  # function
                else:
                    features = extractor.extract(image)  # class

                features = np.asarray(features).flatten()
                feature_list.append(features)

            except Exception as e:
                print(f"Error in {extractor}: {e}")

        if len(feature_list) == 0:
            raise ValueError("No features extracted!")

        return np.concatenate(feature_list)