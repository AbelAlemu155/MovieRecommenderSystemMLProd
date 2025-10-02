from logger import configure_logging, attach_flask_logger, start_periodic_log_cleaner, LOG_PATH
from flask import Flask
from models.model_factory import RecommenderFactory, RecommenderModelType


app = Flask(__name__)

root_logger = configure_logging(logger_name="recommender")
attach_flask_logger(app, root_logger)

start_periodic_log_cleaner(
    active_log=LOG_PATH,
    backup_count=5,
    interval_seconds=10,
)

@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    app.logger.info(f"Received recommendation request for user ID: {userid}")
    app.logger.info(f"Received recommendation request for user ID: {userid}")

    rf = RecommenderFactory()
    recommender_type = RecommenderModelType.UBCF
    recommendations = []
    response = ""

    app.logger.info(f"Using {recommender_type} for recommendations")

    try:
        recommender = rf.get_recommender(recommender_type)
        recommendations = recommender.recommend(user_id=int(userid))
        response = recommender.format(recommendations)
    except Exception as e:
        app.logger.warning(
            f"Primary model failed to find user recommendations: {e}"
        )
        app.logger.info("Switching to RecommenderModelType.DEEP_LEARNING")
        try:
            default_recommender = rf.get_recommender(
                RecommenderModelType.MATRIX_FACTOR
            )
            recommendations = default_recommender.recommend(user_id=int(userid))
            response = default_recommender.format(recommendations)
        except Exception as fallback_error:
            app.logger.error(
                f"FATAL: Fallback model failed too. Error: {fallback_error}"
            )
            recommendations = []

    app.logger.info(f"Response: {response}")
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8082, threaded=True)
