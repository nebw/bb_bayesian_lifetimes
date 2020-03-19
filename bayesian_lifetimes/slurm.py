
import numpy as np
import pandas as pd


def get_data_for_bee(bee_id, bee_detections, 
                     min_doy, max_doy, use_tagged_date,
                     num_tune=2000, num_draws=1000):
    import itertools
    import pandas as pd
    import numpy as np
    import datetime
    import scipy
    import scipy.stats
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ['THEANO_FLAGS'] = "base_compiledir={}".format(tmpdirname)
        from bayesian_lifetimes.estimator import LifetimeEstimator

        estimator = LifetimeEstimator(min_doy=min_doy, max_doy=max_doy, 
                                      use_tagged_date=use_tagged_date)
        _, trace, detections = estimator.fit(bee_id, bee_detections, 
                                             num_tune=num_tune, 
                                             num_draws=num_draws)

    emerged_doy = scipy.stats.mode(trace['switchpoint_emerged'])[0]
    died_doy = scipy.stats.mode(trace['switchpoint_died'])[0]

    return (bee_id, emerged_doy, died_doy, detections, trace)


def generate_jobs(detections_path, log_detections_threshold=9.5):
    detections = pd.read_parquet(detections_path)

    log_dets = np.log(detections.groupby('bee_id').max()['count'])
    log_dets = log_dets[log_dets > log_detections_threshold]
    detections = detections[detections.bee_id.isin(set(log_dets.index.values))]

    for bee_id, bee_detections in detections.groupby('bee_id'):
        yield dict(bee_id=bee_id, bee_detections=bee_detections)