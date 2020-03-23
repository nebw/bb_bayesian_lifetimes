import datetime
import numpy as np
import pandas as pd
import scipy
import scipy.stats

import pymc3 as pm
from bb_utils.ids import BeesbookID
from bb_utils.meta import BeeMetaInfo


class LifetimeEstimator:
    mu_days_alive = 21
    sigma_days_alive = 25
    p_tagged = 0.75

    def __init__(
        self,
        min_doy=201,
        max_doy=263,
        pad_days_after=40,
        use_tagged_date=True,
        min_detections=np.expm1(8),
        max_detections=np.expm1(12),
        dead_rate_beta=25,
        p_hatchdates=None,
    ):
        self.min_doy = min_doy
        self.max_doy = max_doy + pad_days_after
        self.pad_days_after = pad_days_after

        self.min_detections = min_detections
        self.max_detections = max_detections
        self.dead_rate_beta = dead_rate_beta

        all_doys = np.arange(self.min_doy, self.max_doy).astype(np.float)
        self.all_doys = pd.DataFrame(all_doys, columns=["doy"])

        self.use_tagged_date = use_tagged_date
        if use_tagged_date:
            self.meta = BeeMetaInfo()

        self.p_hatchdates = p_hatchdates

    def fit(self, bee_id, bee_detections, num_tune=1000, num_draws=1000, progress=False):
        bee_detections = bee_detections.merge(self.all_doys, how="outer")

        bee_detections.fillna(0, inplace=True)
        bee_detections.sort_values("doy", inplace=True)
        bee_detections.bee_id = bee_id

        num_detections = bee_detections["count"].values
        num_detections -= self.min_detections
        num_detections = np.clip(num_detections, 0, self.max_detections)
        num_detections = num_detections / self.max_detections

        days = np.array(list(range(num_detections.shape[0]))).astype(np.float64)
        num_detections = num_detections.astype(np.float32)

        if self.use_tagged_date:
            hatchdate = self.meta.get_hatchdate(BeesbookID.from_ferwar(bee_id))
            tagged_doy = (hatchdate - datetime.datetime(hatchdate.year, 1, 1)).days
            tagged_dor = tagged_doy - self.min_doy + 1
            tagged_day = np.clip(tagged_dor, 0, np.inf)
            tagged_day_unclipped = tagged_dor
            num_days_clipped = np.abs(tagged_day - tagged_day_unclipped)

            if np.isnan(tagged_day):
                return None
        else:
            num_days_clipped = 0

        model = pm.Model()
        with model:
            p_emergence = np.ones(len(days))
            p_emergence[-self.pad_days_after :] = 0.0

            if self.p_hatchdates is not None:
                p_emergence *= 0.001
                p_emergence[list(map(int, self.p_hatchdates))] = 1.0
            p_emergence /= p_emergence.sum()

            if self.use_tagged_date:
                p_emergence[int(tagged_day)] = self.p_tagged
                p_emergence /= p_emergence.sum()

            p_days_alive = scipy.stats.norm.pdf(
                np.arange(0, len(days)), self.mu_days_alive, self.sigma_days_alive
            )
            p_days_alive /= p_days_alive.sum()

            switchpoint_emerged = pm.Categorical("switchpoint_emerged", p=p_emergence)
            days_alive = pm.Categorical("days_alive", p=p_days_alive)
            switchpoint_died = pm.Deterministic(
                "switchpoint_died", switchpoint_emerged + days_alive - num_days_clipped
            )

            threshold = pm.Beta("dead_rate", alpha=1, beta=self.dead_rate_beta)
            probability_higher = pm.Beta("probability_higher", alpha=5, beta=1)
            probability_lower = pm.Beta("probability_lower", alpha=1, beta=5)

            rate = pm.math.switch(
                (days >= switchpoint_emerged) & (days <= switchpoint_died),
                probability_higher,
                probability_lower,
            )

            _ = pm.Bernoulli("detections", rate, observed=num_detections > threshold)

            trace = pm.sample(tune=num_tune, draws=num_draws, progressbar=progress)

        return model, trace, num_detections
