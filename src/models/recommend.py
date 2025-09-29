import argparse
from models.model_factory import RecommenderFactory, RecommenderModelType

def model_type_parser(arg: str) -> RecommenderModelType:
    """
    Custom parser function for argparse to convert a string argument 
    to the corresponding RecommenderModelType enum member.
    """
    upper_arg = arg.upper()
    try:
        RecommenderModelType(upper_arg)
        return upper_arg
    except ValueError:
        valid_choices = [e.value for e in RecommenderModelType]
        raise argparse.ArgumentTypeError(
            f"Invalid model type: '{arg}'. Must be one of {valid_choices}"
        )


def recommend_with_model(model_type: RecommenderModelType, user_id: str):
    """Recommend a list of movies given a user id."""

    print(f"\nRecommending with: {model_type.name} ({model_type.value})")
    try:
        user_id_int = int(user_id)
    except ValueError:
        print(f"Error: user_id must be an integer, got '{user_id}'.")
        return
    recommender = RecommenderFactory.get_recommender(model_type)

    try:
        recommendations = recommender.recommend(user_id_int)
        print(f"Recommendations for user {user_id}: {recommendations}")
        return recommendations
        
    except Exception as e:
        print(f"Recommendation failed for user {user_id}: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get a list of movie recommendations using a specific model"
    )

    valid_model_values = [e.value for e in RecommenderModelType]

    parser.add_argument(
        "model_type",
        type=model_type_parser,
        choices=valid_model_values,
        help=f"The type of model to recommend with. Choose from: {', '.join(valid_model_values)}"
    )

    parser.add_argument(
        "user_id",
        type=str,
        help=f"The numerical id of the user to generate recommendations for."
    )

    args = parser.parse_args()
    
    model_enum = RecommenderModelType(args.model_type) 

    recommend_with_model(model_enum, args.user_id)