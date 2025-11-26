import time
import logging
from prometheus_client import Gauge, start_http_server
from online_evaluation.online_evaluation import compute_online_measure
from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
from scipy.stats import ttest_ind

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metric
MODEL_QUALITY_GAUGE = Gauge('model_quality_success_rate', 'Proportion of successful recommendations')

TREATMENT_QUALITY_GAUGE= Gauge('model_quality_treatment_group', 'Proportion of successful recommendations-Treatment')

CONTROL_QUALITY_GAUGE= Gauge('model_quality_control_group', 'Proportion of successful recommendations-Control')

T_STAT_GAUGE=  Gauge('t_stat_ab', 't-statisics difference A/B testing')
P_STAT_GAUGE=  Gauge('p_value_ab', 'p-value A/B testing')


def monitor_model_quality():
    # Start Prometheus HTTP server on port 8001
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001")
    logger.info("Model quality monitoring started")

    while True:
        try:
            logger.info("Computing model quality...")
            success_proportion = compute_online_measure()
            MODEL_QUALITY_GAUGE.set(success_proportion)

            success_proportion_treatment= compute_online_measure(treatment_group=True)
            TREATMENT_QUALITY_GAUGE.set(success_proportion_treatment)

            success_proportion_control= compute_online_measure(treatment_group=False)
            CONTROL_QUALITY_GAUGE.set(success_proportion_control)

            t_stat, p_value= compute_tstat()
            T_STAT_GAUGE.set(t_stat)

            P_STAT_GAUGE.set(p_value)
            logger.info(f"Model quality metric updated: {success_proportion:.4f}")
            logger.info(f"Treatment group quality metric updated: {success_proportion_treatment}")
            logger.info(f"Control group quality metric updated: {success_proportion_control}")
            logger.info(f"t-statistics A/B testing: {t_stat}")
            logger.info(f"pvalue A/B testing: {p_value}")
            
            
        except Exception as e:
            logger.error(f"Error computing model quality: {e}")

        logger.info("Sleeping for 30 seconds...")
        time.sleep(30)

def compute_tstat():
    
    # example env for your container
    PROMETHEUS_URL='http://host.docker.internal:9090'


    prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

    end = datetime.utcnow()
    start = end - timedelta(hours=12)

    treatment = prom.get_metric_range_data(
        "model_quality_treatment_group",
        start_time=start,
        end_time=end
    )

    control = prom.get_metric_range_data(
        "model_quality_control_group",
        start_time=start,
        end_time=end
    )
    # extract values
    t_measures = [float(v[1]) for v in treatment[0]['values']]
    c_measures = [float(v[1]) for v in control[0]['values']]
    
    len_diff = len(t_measures) - len(c_measures)

    if len_diff > 0:
        # control is smaller → pad with zeros
        c_measures += [0.0] * len_diff
    elif len_diff < 0:
        # treatment is smaller → pad with zeros
        t_measures += [0.0] * (-len_diff)

    # now both have same length
    t_stat, p_value = ttest_ind(t_measures, c_measures, equal_var=False)
    t_stat, p_value = ttest_ind(t_measures, c_measures, equal_var=False)

    return t_stat, p_value

if __name__ == '__main__':
    monitor_model_quality()
