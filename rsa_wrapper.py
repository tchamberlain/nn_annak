import random
import numpy as np
import pandas as pd
from make_behavioral_mtx import make_behavioral_mtx, get_behav
from make_nodewise_isc_mtx import make_nodewise_isc_mtx, get_all_timeseries_mtx
from get_rsa_permutation_results import get_rsa_permutation_results



def run_rsa(subs,
            ds,
            task,
            run,
            function,
            n_perms,
            use_controls):

    # grab behavioral similarity
    behav_vct = make_behavioral_mtx(get_behav(subs, task, run, ds), function)

    # grab brain similarity
    if run=='movieIS_avg_mats':
        ts_1 = get_all_timeseries_mtx(subs,
                                        'movieIS_session1')
        brain_vct_1 = make_nodewise_isc_mtx(ts_1)

        ts_2 = get_all_timeseries_mtx(subs,
                                        'movieIS_session2')
        brain_vct_2 = make_nodewise_isc_mtx(ts_2)
        brain_vct = np.mean([brain_vct_1, brain_vct_2], axis = 0)
    else:
        ts = get_all_timeseries_mtx(subs,
                                    run)
        brain_vct = make_nodewise_isc_mtx(ts)

    # run the rsa, save the results in a df
    res = get_rsa_permutation_results(brain_vct,
                                     behav_vct,
                                     subs,
                                     run,
                                     ds,
                                     run_age=(task=='age'),
                                     use_controls=use_controls,
                                     n_perms=n_perms)

    df = pd.DataFrame({'r_val': res[:, 0],
                       'p_val': res[:, 1], 
                       'null_r_vals': res[:, 2],})
    df['run'] = run
    df['task'] = task
    df['function'] = function
    df['n_perms'] = n_perms
    df['use_controls'] = use_controls
    df['dataset'] = ds

    return df

def rsa_wrapper(ds,
                task,
                run,
                function,
                n_perms,
                use_controls,
                run_split_half):
    use_controls_title = ''
    if use_controls:
        use_controls_title = 'control'
    subs = np.loadtxt(f'subs/{ds}/{run}_subs.txt', dtype=str)

    if run_split_half:
        random.Random(1).shuffle(subs)
        half = int(len(subs)/2)
        subs1, subs2= subs[0:half], subs[half:]

        df1 = run_rsa(subs1,
                    ds,
                    task,
                    run,
                    function,
                    n_perms,
                    use_controls=use_controls)
        df1['half'] = 1
        df1['null_r_vals'] = np.nan
        df2 = run_rsa(subs2,
                    ds,
                    task,
                    run,
                    function,
                    n_perms,
                    use_controls=use_controls)
        df2['null_r_vals'] = np.nan
        df2['half'] = 2

        df = pd.concat([df1, df2])
        out_folder = 'outputs/split_halves'
    else:
        out_folder = 'outputs/'
        df = run_rsa(subs,
                    ds,
                    task,
                    run,
                    function,
                    n_perms,
                    use_controls=use_controls)
    out_file = f'{out_folder}/out_{ds}_{run}_{task}_{function}_{n_perms}_{use_controls_title}.csv'
    print('saving to: ', out_file)
    df.to_csv(out_file)
