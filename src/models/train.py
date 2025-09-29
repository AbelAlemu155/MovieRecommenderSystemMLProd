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


def train_model(model_type: RecommenderModelType):
    """Initializes, trains, and saves a specific model."""

    print(f"\nStarting training for: {model_type.name} ({model_type.value})")
    recommender = RecommenderFactory.get_recommender(model_type)
    recommender.train()
    print(f"Model {model_type.name} trained and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a specific recommender model and save the artifact."
    )

    valid_model_values = [e.value for e in RecommenderModelType]

    parser.add_argument(
        "model_type",
        type=model_type_parser,
        choices=valid_model_values,
        help=f"The type of model to train. Choose from: {', '.join(valid_model_values)}"
    )

    args = parser.parse_args()
    model_enum = RecommenderModelType(args.model_type) 
    train_model(model_enum)