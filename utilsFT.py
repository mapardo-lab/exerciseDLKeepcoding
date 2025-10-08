import statistics
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sentence_transformers import SentenceTransformer
import torchvision.transforms as transforms
import cv2
from PIL import Image

class EmbeddingText(BaseEstimator, TransformerMixin):
    def __init__(self, feature, model_name):
        self.feature = feature
        self.model_name = model_name
    
    def transform(self, df):
        encoder = SentenceTransformer(self.model_name)
        embeddings = encoder.encode(list(df[self.feature]))
        return embeddings
        
class TargetFeature(BaseEstimator, TransformerMixin):
    def __init__(self, col1_name, col2_name):
        self.col1_name = col1_name
        self.col2_name = col2_name
        self.scaler1 = MinMaxScaler()
        self.scaler2 = MinMaxScaler()
        self.threshold = None
    
    def fit(self, X, y=None):
        col1_scaled = self.scaler1.fit_transform(X[[self.col1_name]])
        col2_scaled = self.scaler2.fit_transform(X[[self.col2_name]])
        target = (col1_scaled + col2_scaled) / 2.0
        self.threshold = statistics.median(target)
        return self
    
    def transform(self, X):
        col1_scaled = self.scaler1.transform(X[[self.col1_name]])
        col2_scaled = self.scaler2.transform(X[[self.col2_name]])
        target = (col1_scaled + col2_scaled) / 2.0
        return np.array([1 if x[0] > self.threshold else 0 for x in target])
        
class ImagesResNet18Transform(BaseEstimator, TransformerMixin):
    def __init__(self, image_path):
        self.image_path = image_path
        self.transform_images = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        ])
        
    def transform(self, X):
        image = cv2.imread(X[self.image_path])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_transformed = self.transform_images(pil_image)
        return image_transformed
        
class MultiLabelBinarizerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for MultiLabelBinarizer to work with ColumnTransformer
    """
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
    
    def fit(self, X, y=None):
        X_series = self._safe_squeeze(X)
        self.mlb.fit(X_series)
        return self
    
    def transform(self, X):
        X_series = self._safe_squeeze(X)
        return self.mlb.transform(X_series)
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_

    def _safe_squeeze(self, X):
        """
        Safely convert one column dataframe to pd.Series with squeeze function
        Only squeezes if X is a one-column DataFrame, otherwise raises error.
        """
        # Check if X is a DataFrame
        if not hasattr(X, 'iloc') or not hasattr(X, 'shape'):
            raise ValueError(
                f"Input must be a DataFrame. Got {type(X)} instead. "
                "Make sure this transformer is used in ColumnTransformer with proper column selection."
            )
        
        # Check if DataFrame has exactly one column
        if X.shape[1] != 1:
            raise ValueError(
                f"Expected 1 column, but got {X.shape[1]} columns. "
                "This transformer should only be applied to a single column. "
                "Check your ColumnTransformer configuration."
            )
        
        # Extract the single column
        return X.squeeze()