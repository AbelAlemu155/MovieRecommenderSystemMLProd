import logging
from logging.handlers import RotatingFileHandler
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from models.model_factory import RecommenderFactory, RecommenderModelType
from models.base_model import Recommender
from models.content_based.content_based_rec import NoUserFound

app = Flask(__name__)

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

file_handler = RotatingFileHandler(
    "recommender_service.log", maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)


@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    app.logger.info(f"Received recommendation request for user ID: {userid}")
    app.logger.info(f"Received recommendation request for user ID: {userid}")

    rf = RecommenderFactory()
    recommender_type = RecommenderModelType.COLLABORATIVE_FILTERING
    recommendations = []
    response = ""

    app.logger.info(f"Using {recommender_type} for recommendations")

    try:
        recommender = rf.get_recommender(recommender_type)
        recommendations = recommender.recommend(user_id=int(userid))
        response = recommender.format(recommendations)
    except Exception as e:
        print(e)
        app.logger.warning(
            f"Primary model failed to find user recommendations: {e}"
        )
        app.logger.info("Switching to RecommenderModelType.DEEP_LEARNING")
        try:
            deep_recommender = rf.get_recommender(
                RecommenderModelType.DEEP_LEARNING
            )
            recommendations = deep_recommender.recommend(user_id=int(userid))
            response = deep_recommender.format(recommendations)
        except Exception as fallback_error:
            app.logger.error(
                f"FATAL: Fallback model failed too. Error: {fallback_error}"
            )
            recommendations = []

    app.logger.info(f"Response: {response}")
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8082, threaded=True)
