
from .collaborative_filtering.item_based_filtering import ItemBasedFiltering

from .ubcf.model import UBCFRecommender
from .base_model import Recommender
from enum import Enum


class RecommenderModelType(Enum):
    COLLABORATIVE_FILTERING = "CF"
    CONTENT_BASED = "CB"
    UBCF = "UBCF"
    DEEP_LEARNING = "DL"

class RecommenderFactory:
    """
    Factory class responsible for creating Recommender instances 
    based on a provided model name or ID.
    """
    _MODELS = {
        RecommenderModelType.COLLABORATIVE_FILTERING.value: ItemBasedFiltering,
       
        RecommenderModelType.UBCF.value: UBCFRecommender,
    }

    @staticmethod
    def get_recommender(model_type: RecommenderModelType) -> Recommender:
        """
        Creates and returns an instance of the specified Recommender model.

        Args:
            model_type: The type of the recommender model using the RecommenderModelType enum

        Returns:
            An instance of a concrete Recommender.

        Raises:
            ValueError: If the model name is not recognized.
        """
        model_key = model_type.value
        if model_key in RecommenderFactory._MODELS:
            return RecommenderFactory._MODELS[model_key]()
        else:
            return CollaborativeFilteringRecommender()
