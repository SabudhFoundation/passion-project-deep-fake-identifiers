import numpy as np

from ..utils.lbp_features import LBPExtractor
from ..utils.glcm_features import GLCMExtractor
from ..utils.fft_features import FFTExtractor


class FeatureBuilder:
    def __init__(self, use_lbp=True, use_glcm=True, use_fft=True):
        self.extractors = []

        if use_lbp:
            self.extractors.append(LBPExtractor())

        if use_glcm:
            self.extractors.append(GLCMExtractor())

        if use_fft:
            self.extractors.append(FFTExtractor())

        if len(self.extractors) == 0:
            raise ValueError("No feature extractors selected!")

    def extract_features(self, image):
        feature_list = []

        for extractor in self.extractors:
            try:
                features = extractor.extract(image)
                features = np.asarray(features).flatten()
                feature_list.append(features)

            except Exception as e:
                print(f"Error in {extractor.__class__.__name__}: {e}")

        if len(feature_list) == 0:
            raise ValueError("No features extracted!")

        return np.concatenate(feature_list)