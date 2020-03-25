import datetime

import numpy as np
import pandas as pd

from bb_utils import meta
from bb_utils.ids import BeesbookID


def get_data_for_bee(
    bee_id,
    bee_detections,
    min_doy,
    max_doy,
    use_tagged_date,
    num_tune=2000,
    num_draws=1000,
    min_detections=20,
    max_detections=3000,
    dead_rate_beta=50,
    p_hatchdates=None,
):
    import scipy
    import scipy.stats
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.environ["THEANO_FLAGS"] = f"base_compiledir={tmpdirname}"
        from bayesian_lifetimes.estimator import LifetimeEstimator

        estimator = LifetimeEstimator(
            min_doy=min_doy,
            max_doy=max_doy,
            use_tagged_date=use_tagged_date,
            min_detections=min_detections,
            max_detections=max_detections,
            dead_rate_beta=dead_rate_beta,
            p_hatchdates=p_hatchdates,
        )
        _, trace, detections = estimator.fit(
            bee_id, bee_detections, num_tune=num_tune, num_draws=num_draws
        )

    emerged_doy = scipy.stats.mode(trace["switchpoint_emerged"])[0]
    died_doy = scipy.stats.mode(trace["switchpoint_died"])[0]

    return (bee_id, emerged_doy, died_doy, detections, trace)


def datetime_to_doy(dt):
    return (dt - datetime.datetime(year=dt.year, month=1, day=1, tzinfo=dt.tzinfo)).days + 1


def generate_jobs(
    detections_path, log_detections_threshold=9.5, use_hatchdates=False, p_hatchdates=None, **kwargs
):
    detections = pd.read_parquet(detections_path)
    m = meta.BeeMetaInfo()

    if p_hatchdates == "bb19":
        assert "min_doy" in kwargs and "max_doy" in kwargs
        idmapping = m.idmapping
        idmapping["doy"] = idmapping["date"].apply(datetime_to_doy)
        idmapping.sort_values("doy", inplace=True)

    if use_hatchdates:
        valid_bees = m.hatchdates[np.logical_not(pd.isna(m.hatchdates.hatchdate))]
        bees = list(map(BeesbookID.from_dec_12, valid_bees.dec12))
        valid_ids = detections.bee_id.isin(set(map(lambda bee_id: bee_id.as_ferwar(), bees)))
        detections = detections[valid_ids]

    log_dets = np.log(detections.groupby("bee_id").max()["count"])
    log_dets = log_dets[log_dets > log_detections_threshold]
    detections = detections[detections.bee_id.isin(set(log_dets.index.values))]

    for bee_id, bee_detections in detections.groupby("bee_id"):
        if p_hatchdates == "bb19":
            earliest_doy = idmapping[idmapping.mapped_id == bee_id].doy.iloc[0]
            tag_id = idmapping[idmapping.mapped_id == bee_id].bee_id.iloc[0]
            later_reuses = idmapping[idmapping.bee_id == tag_id]
            later_reuses = later_reuses[later_reuses.doy > earliest_doy]
            latest_doy = later_reuses.doy.iloc[0] if len(later_reuses) > 0 else kwargs["max_doy"]
            kwargs["p_hatchdates"] = np.arange(earliest_doy, latest_doy) - kwargs["min_doy"]

        yield dict(
            bee_id=bee_id, bee_detections=bee_detections, use_tagged_date=use_hatchdates, **kwargs
        )
