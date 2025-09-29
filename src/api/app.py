from flask import Flask
from models.model_factory import RecommenderFactory, RecommenderModelType
from models.base_model import Recommender
from models.content_based.content_based_rec import NoUserFound


app = Flask(__name__)


@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    print(f"Received recommendation request for user ID: {userid}")

    rf = RecommenderFactory()
    recommender_type = RecommenderModelType.DEEP_LEARNING
    recommendations = []
    response = ""

    try:
        recommender = rf.get_recommender(recommender_type)
        recommendations = recommender.recommend(user_id=int(userid))
        response = recommender.format(recommendations)
    except Exception as e:
        print(f"Primary model failed to find user recommendations: {e}")
        try:
            deep_recommender = rf.get_recommender(RecommenderModelType.DEEP_LEARNING)
            recommendations = deep_recommender.recommend(user_id=int(userid))
            response = deep_recommender.format(recommendations)
        except Exception as fallback_error:
            print(f"FATAL: Fallback model failed too. Error: {fallback_error}")
            recommendations = []
    return response

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8082, threaded=True)