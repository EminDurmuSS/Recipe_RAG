import torch
import gc
import logging
from typing import Tuple

from pykeen.models import Model
from pykeen.triples import TriplesFactory

from utils import *

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class for managing KGE model and triples factory.
    Ensures model is loaded only once and shared across requests.
    """
    _instance = None
    _model = None
    _triples_factory = None
    _is_loading = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logger.info("Creating new ModelManager instance")
            cls._instance = ModelManager()
        return cls._instance
    
    def get_model_and_triples(self, MODEL_PATH, TRIPLES_PATH) -> Tuple[Model, TriplesFactory]:
        """
        Get the model and triples factory. If they're not loaded yet,
        load them. Thread-safe to avoid duplicate loading.
        """
        # Check if already loaded
        if self._model is not None and self._triples_factory is not None:
            return self._model, self._triples_factory
        
        # Prevent concurrent loading
        if self._is_loading:
            logger.info("Model is already being loaded by another thread, waiting...")
            while self._is_loading:
                pass  # Simple spin lock
            return self._model, self._triples_factory
        
        try:
            self._is_loading = True
            logger.info("Loading model and triples...")
            
            # Memory cleanup before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model and triples
            with torch.no_grad():  # Prevent memory leaks from gradients
                self._model = load_kge_model(MODEL_PATH).eval()
                self._triples_factory = get_triples_factory(TRIPLES_PATH)
            
            logger.info("Model and triples loaded successfully")
            return self._model, self._triples_factory
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
        
        finally:
            self._is_loading = False
            # Memory cleanup after loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Create singleton instance at module load time
model_manager = ModelManager.get_instance()