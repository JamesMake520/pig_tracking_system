"""
ReID Feature Extractor wrapper for DeepOC-SORT
Uses deep_sort_realtime's embedding models
"""
import numpy as np
import torch
import cv2


class ReIDFeatureExtractor:
    """
    Wrapper for ReID feature extraction using deep_sort_realtime
    """

    def __init__(self, model_type='mobilenet', use_cuda=True):
        """
        Initialize ReID feature extractor

        Args:
            model_type: Type of embedding model ('mobilenet', 'torchreid', 'clip_RN50', etc.)
            use_cuda: Whether to use CUDA for inference
        """
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

        # Import and initialize embedder from deep_sort_realtime
        try:
            from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder
            print(f"Initializing ReID feature extractor: {model_type}")
            # MobileNetv2_Embedder parameters: model_wts_path, half, max_batch_size, bgr, gpu
            use_gpu = (self.device == 'cuda')
            self.embedder = MobileNetv2_Embedder(
                model_wts_path=None,  # Use built-in weights
                half=False,  # Don't use FP16
                max_batch_size=16,
                bgr=True,  # Input is BGR format
                gpu=use_gpu
            )
            print(f"ReID model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load ReID model: {e}")
            print("DeepOC-SORT will fallback to motion-only tracking")
            self.embedder = None

    def __call__(self, crops):
        """
        Extract features from image crops

        Args:
            crops: List of image crops (BGR format) or single crop

        Returns:
            features: numpy array of shape (N, D) where D is feature dimension
        """
        if self.embedder is None:
            return None

        # Handle single crop
        if not isinstance(crops, list):
            crops = [crops]

        if len(crops) == 0:
            return None

        try:
            # Preprocess crops
            processed_crops = []
            for crop in crops:
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    # Invalid crop, use dummy placeholder
                    processed_crops.append(np.zeros((128, 64, 3), dtype=np.uint8))
                else:
                    # Resize to standard size if needed
                    if crop.shape[0] < 20 or crop.shape[1] < 20:
                        crop = cv2.resize(crop, (64, 128))
                    processed_crops.append(crop)

            # Extract features using embedder
            features = self.embedder.predict(processed_crops)

            # Convert to numpy if tensor
            if torch.is_tensor(features):
                features = features.cpu().numpy()

            return np.array(features)

        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return None

    def is_available(self):
        """Check if embedder is available"""
        return self.embedder is not None


class DummyFeatureExtractor:
    """
    Dummy feature extractor that returns None
    Used when ReID model is not available or disabled
    """

    def __call__(self, crops):
        return None

    def is_available(self):
        return False
