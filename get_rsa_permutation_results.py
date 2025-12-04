import os
import numpy as np
import pandas as pd
import mantel_with_rand_seeds
from statsmodels.formula.api import ols
from scipy import stats
from tqdm import tqdm
from make_behavioral_mtx import get_behav, make_behavioral_mtx
from joblib import Parallel, delayed


def get_residuals(vals,
                  nanmask,
                  subs,
                  run,
                  ds,
                  use_motion_control=True,
                  run_interaction=True,
                  run_age=False):

        age = get_behav(subs, 'age', run, ds)
        motion = get_behav(subs, 'motion', run, ds)
        sex = get_behav(subs, 'sex', run, ds)
        df_tmp = pd.DataFrame({'vals':stats.zscore(vals[nanmask]),

                               'age_mean': make_behavioral_mtx(age, 'nn')[nanmask],
                               'age_diff': make_behavioral_mtx(age, 'annak')[nanmask],

                               'motion_diff': make_behavioral_mtx(motion, 'nn')[nanmask],
                               'motion_mean': make_behavioral_mtx(motion, 'annak')[nanmask],
 
                               'sex_diff': make_behavioral_mtx(sex, 'nn')[nanmask], })
        if run_age:
            formula = 'vals ~ sex_diff'
        elif run_interaction:
            formula = 'vals ~ age_mean*sex_diff*age_diff'
        else:
            formula = 'vals ~ age_mean + age_diff + sex_diff'

        if use_motion_control:
            formula+='+ motion_mean + motion_diff'

        model = ols(formula,
                    data=df_tmp)
        results = model.fit()
        resids = np.array(results.resid)
        return resids


def _rsa_single_node(node,
                     brain_vct,
                     behav_vct,
                     subs,
                     run,
                     ds,
                     n_perms,
                     method,
                     run_age,
                     use_controls, 
                     use_motion_control=True):
    node_vct = brain_vct[:, node]
    nanmask = ~np.isnan(node_vct)

    node_vct_tmp = node_vct[nanmask]
    behav_vct_tmp = behav_vct[nanmask]
    
    if use_controls:
        node_vct_tmp = get_residuals(node_vct, nanmask, subs, use_motion_control=use_motion_control, 
                                     run=run, ds=ds, run_age=run_age)
        behav_vct_tmp = get_residuals(behav_vct, nanmask, subs, use_motion_control=use_motion_control, 
                                      run=run, ds=ds, run_age=run_age)


    res = mantel_with_rand_seeds.test(node_vct_tmp,
                                      behav_vct_tmp,
                                      method=method,
                                      perms=n_perms,
                                      ignore_nans=True)
    return [res.r, res.p, list(res.correlations)]



def get_rsa_permutation_results(brain_vct,
                                behav_vct,
                                subs,
                                run,
                                ds,
                                n_perms=1000,
                                method='spearman',
                                run_age=False,
                                use_controls=True, 
                                use_motion_control=True):

    n_nodes = brain_vct.shape[1]

    # Parallel across nodes
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(_rsa_single_node)(
            node,
            brain_vct,
            behav_vct,
            subs,
            run,
            ds,
            n_perms,
            method,
            run_age=run_age,
            use_controls=use_controls, 
            use_motion_control=use_motion_control,
        )
        for node in range(n_nodes)
    )

    return np.array(results, dtype=object)
