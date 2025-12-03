import numpy as np
import pandas as pd

def make_behavioral_mtx(behav_vct, model):
    """
    Takes a vector of behavioral scores (one per subject) and returns
    the vectorized upper triangle of a similarity matrix constructed using one formulation of the "AnnaK" principle
    (i.e., high-high pairs are most alike, low-low pairs are most dissimilar, and high-low pairs show intermediate similarity).

    """

    # Get dims
    n_subs = len(behav_vct)

    # Initialize output
    mtx = np.zeros((n_subs,n_subs))

    for i in range(n_subs):
        for j in range(n_subs):
            if model=='annak':
                mtx[i,j] = (behav_vct[i] + behav_vct[j])/2
            elif model=='nn':
                mtx[i,j] = (-1)*abs(behav_vct[i] - behav_vct[j])
            elif model=='nearavg':
                avg = np.mean(behav_vct)
                mtx[i,j] = abs(behav_vct[i] - avg)+abs(behav_vct[j] - avg)


    # Return just the upper triangle
    vct = mtx[np.triu_indices(mtx.shape[0], k=1)]
    return vct


def get_behav(subs, task, run, ds):
    behav = pd.read_csv(f'./data/{ds}_behav.csv', low_memory=False)
    behav['subject'] = behav['subject'].astype(str)
    if task =='motion':
        if run =='movieIS_avg_mats':
            task = f'motion_movieIS_avg'
        else:
            task = f'motion_{run}'
    vals = []
    for sub in subs:
        vals.append(behav[behav['subject']==sub][task].values[0])
    return np.array(vals)
