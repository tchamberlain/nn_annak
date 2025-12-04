import numpy as np
import pandas as pd

def get_all_timeseries_mtx(subs, run):
    """
    Takes a list of subjects and a run, and returns a matrix that is nodes x TRs x subs
    """
    to_concat  = []
    for sub in subs:
        x = np.loadtxt(f'./data/ts_files/{sub}_{run}.txt')
        to_concat.append(x)

    res = np.array(to_concat)
    return np.transpose(res, [1, 2, 0])

def make_nodewise_isc_mtx(D, tr_range=None, include_nans=True, vectorized=True):
    # If a TR range was given, restrict data matrix to just those timepoints
    if tr_range:
        D = D[:, tr_range[0]:tr_range[1], :]

    # Get dims
    n_nodes, n_trs, n_subs = D.shape

    # Initalize result
    out = np.zeros((n_subs, n_subs, n_nodes))

    for n in range(n_nodes):
        df = pd.DataFrame(D[n,:,:], range(n_trs))
        out[:,:,n] = df.corr().values

    vect_out = out[np.triu_indices(out.shape[0], k=1)]
    if vectorized:
        return vect_out
    return out
