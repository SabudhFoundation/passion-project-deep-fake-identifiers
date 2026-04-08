import numpy as np
import pickle

class FeatureNormalizer:
    def __init__(self, method="standard"):
        self.min = None
        self.max = None
        
        self.mean = None
        self.std = None

        self.median = None
        self.iqr = None
    
    # ====================================
    # 1. Min_Max Normalization
    # ====================================
    def fit(self,x):
        self.min = np.min(x, axis = 0)
        self.max = np.max(x, axis = 0)
    
    def transform_minmax(self,x):
        return (x-self.min)/(self.max-self.min)

    
    # ======================================
    # 2. Standardization(z-score)
    # ======================================
    def fit_standard(self,x):
        self.mean = np.mean(x,axis = 0)
        self.std = np.std(x,axis = 0)+1e-8

    def transform_standard(self,x):
        self.fit_standard(x)
        return self.transform_standard(x)


    # =====================================
    # 3. ROBUST SCALING (MEDIAN + IQR)
    # =====================================
    def fit_robust(self, X):
        self.median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr = (q3 - q1) + 1e-8

    def transform_robust(self, X):
        return (X - self.median) / self.iqr

    def fit_transform_robust(self, X):
        self.fit_robust(X)
        return self.transform_robust(X)


    # ===================================
    # Save/Load
    # ==================================
    def save(self,filepath):
        with open(filepath,"wb") as f:
            pickle.dump(self.__dict__,f)
    

    def load(self,filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.__dict__.update(data)