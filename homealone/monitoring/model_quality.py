import time
import logging
from prometheus_client import Gauge
from online_evaluation.online_evaluation import compute_online_measure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metric
MODEL_QUALITY_GAUGE = Gauge('model_quality_success_rate', 'Proportion of successful recommendations')

def monitor_model_quality():
    logger.info("Model quality monitoring started")

    while True:
        try:
            logger.info("Computing model quality...")
            success_proportion = compute_online_measure()
            MODEL_QUALITY_GAUGE.set(success_proportion)
            logger.info(f"Model quality metric updated: {success_proportion:.4f}")
        except Exception as e:
            logger.error(f"Error computing model quality: {e}")

        logger.info("Sleeping for 30 seconds...")
        time.sleep(30)

if __name__ == '__main__':
    monitor_model_quality()
