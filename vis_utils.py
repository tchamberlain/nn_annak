import pandas as pd
import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats import multitest
import nibabel as nib
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import itertools
from nilearn.plotting import plot_connectome, view_connectome, view_markers, plot_roi
from nilearn import plotting
import matplotlib.image as mpimg

output_dir = './outputs'


def load_up_result_by_task(n_perms=10000,
                            use_controls=True,
                            no_motion_control=False,
                            run_split_half_opts=[True, False], 
                            use_motion_control=True,
                           ds='hbn',
                           run='movieDM',
                           tasks=['listsort']):
    
    to_concat  = []
    functions = ['annak', 'nn']

    for ds in [ds]:
        for task in tasks:
            for function in functions:
                for run in [run]:
                    for run_split_half in run_split_half_opts:
                        out_folder = f'{output_dir}/'
                        if run_split_half:
                            out_folder = f'{output_dir}/split_halves'
                        for use_controls in [use_controls]:
                            use_controls_title = ''
                            if use_controls:
                                use_controls_title = 'control'
                            elif no_motion_control:
                                use_controls_title = 'age_sex_no_motion_control'
                            path = f'{out_folder}/out_{ds}_{run}_{task}_{function}_{n_perms}_{use_controls_title}.csv'

                            df_tmp = pd.read_csv(path)
                            if run_split_half==False:
                                df_tmp['half'] = 'all'
                            to_concat.append(df_tmp)
    df = pd.concat(to_concat)
    df['node'] = df['Unnamed: 0']
    return df


def color_rois(values):

    """
    This function assumes you are passing a vector "values" with the same length as the number of nodes in the atlas.
    """

    shen268 = nib.load("./data/shen_2mm_268_parcellation.nii.gz")
    shen268_data = shen268.get_fdata()
    img = np.zeros(shen268_data.shape)

    for roi in range(len(values)):
        itemindex = np.where(shen268_data==roi+1) # find voxels in this node (add 1 to account for zero-indexing)
        img[itemindex] = values[roi] # color them by the desired value

    affine = shen268.affine
    img_nii = nib.Nifti1Image(img, affine)

    return img_nii

def make_glass_brain(nodes,
                 sig_nodes,
                 r_vals_input,
                 title,
                 sig_nodes_corrected=[],
                 cmap = 'Spectral_r',
                 ax=None,
                 vmax=0.2):

    # only color in the nodes passed in
    # if all 268 are passsed in, all nodes are colored in
    r_vals = np.full([268], np.nan)
    for index, node in enumerate(nodes):
        r_vals[node] = r_vals_input[index]

    the_min = vmax*-1
    the_max = vmax
    display1 = plotting.plot_glass_brain(color_rois(r_vals_input),
                                         cmap=cmap,
                                         colorbar=False,
                                         plot_abs=False,
                                         vmin=the_min,
#                                          threshold=.05,
                                         vmax=the_max,
                                         symmetric_cbar=True,
                                         display_mode='lr',
                                         axes=ax,
                                        )

    # add gray outlines for nodes that survive the pthreshold
    for node in sig_nodes:
        nodes = [0 for i in range(268)]
        nodes[node] = r_vals[node]
        display1.add_contours(color_rois(nodes), linewidths=.1, colors=['#3e3f40'])

    # add gray outlines for nodes that survive correction for multiple comparisons
    for node in sig_nodes_corrected:
        nodes = [0 for i in range(268)]
        nodes[node] = r_vals[node]
        display1.add_contours(color_rois(nodes), linewidths=.2, colors=['black'])



def compare_outputs_split_half(ds,func,task, df,  run='movieDM',  use_controls=True, print_res=False):
    # black for actual

    dfs = df[(df['dataset']==ds)&
             (df['function']==func)&
              (df['run']==run)&
                (df['use_controls']==use_controls)&
               (df['task']==task)]

    h1 = dfs[dfs['half']==1]
    h2 = dfs[dfs['half']==2]

    p1 = h1.p_val.values
    p2 = h2.p_val.values

    r1 = h1.r_val.values
    r2 = h2.r_val.values

    alpha1  = .0135
    n_nodes = r1.shape[0]

    surv1 = p1 < alpha1 # does it survive in result 1?
    surv2 = p2< alpha1 # does it survive in result 2?
    signs = np.sign(r1)==np.sign(r2) # are the signs the same?
    surv_bool = surv1 & surv2 & signs
    surv_nodes1  = np.array(range(n_nodes))[surv_bool]

    alpha1  = .05
    surv1 = p1 < alpha1 # does it survive in result 1?
    surv2 = p2< alpha1 # does it survive in result 2?
    signs = np.sign(r1)==np.sign(r2) # are the signs the same?
    surv_bool2 = surv1 & surv2 & signs
    surv_nodes2  = np.array(range(n_nodes))[surv_bool2]
    hue = []
    for x in range(len(r1)):

        hue.append('silver')

    if print_res:
        print('\n', task)
        print( 'num surviving nodes .05', surv1.sum(), surv2.sum())
        print( 'num surviving nodes .05', surv_nodes2.shape[0])
        print('num surviving nodes  .0135 ', surv_nodes1.shape[0])

    return r1, r2, hue


def compare_outputs(ds,func,task, df, half='all', print_res=False, use_controls=True):
        dfs = df[(df['dataset']==ds)&
                 (df['function']==func)&
                    (df['use_controls']==use_controls)&
                   (df['task']==task)]
        h1 = dfs[dfs['half']==half]

        p1 = h1.p_val.values

        r1 = h1.r_val.values

        alpha1  = .05

        surv1 = p1 < alpha1 # does it survive in result 1?
        surv2 = multitest.fdrcorrection(h1.p_val, alpha=0.05, method='indep', is_sorted=False)[0]

        surv_nodes1  = np.array(range(268))[surv1]
        surv_nodes2  = np.array(range(268))[surv2]

        return surv_nodes1, surv_nodes2, r1

def make_scatterplot(r1, r2, hue, ax, vmax=.2, x_ticks=False):
    corr, _ = stats.pearsonr(r1, r2)
    g = sns.regplot(x=r1, y=r2,ax = ax, color='black',marker='D', 
    scatter_kws={'color': 'silver','alpha':0.6,'s':6,  'marker':'D'})
    ax.axhline(y=0, linewidth=2, color='silver')
    ax.axvline(x=0, linewidth=2, color='silver')
    g.set_xlim([-vmax, vmax])
    g.set_ylim([-vmax, vmax])
    

    g.text(.03, .03, f'r = {round(corr, 2)}', fontsize=20, color='gray', transform=ax.transAxes, weight='bold')

    ax.set_xticks([-1*vmax , 0. ,  vmax])
    ax.set_yticks([-1*vmax , 0. ,  vmax])
    ax.tick_params(axis='y', which='major', labelsize=15, color='gray')
    x_label_size = 0
    if x_ticks:
        x_label_size = 15
        ax.set_xlabel('Cohort 1 RSA r-value')
    
    ax.set_ylabel('Cohort 2 RSA r-value')
    ax.tick_params(axis='x', which='major', labelsize=x_label_size,  color='gray')
    import matplotlib
    matplotlib.rc('axes',edgecolor='gray')

    ax.set_box_aspect(1)
    return ax

def make_glass_brain_dots(nodes,
                 sig_nodes,
                 r_vals_input,
                 display_mode='lr',
                 sig_nodes_corrected=[],
                 cmap = 'Spectral_r',
                 ax=None,
                 vmax=0.2):
    
    
    # only color in the nodes passed in
    # if all 268 are passsed in, all nodes are colored in
    r_vals = np.full([268], np.nan)
    for index, node in enumerate(nodes):
        r_vals[node] = r_vals_input[index]


    the_min = vmax*-1
    the_max = vmax
    
    display1 = plotting.plot_glass_brain(color_rois(r_vals_input),
                                         cmap=cmap,
                                         colorbar=False,
                                         plot_abs=False,
                                         vmin=the_min,
                                         vmax=the_max,
                                         alpha=.7,
                                        #  alpha=1,
                                         symmetric_cbar=True,
                                         display_mode=display_mode,
                                          annotate=False,
                                         axes=ax,
                                         title = '',
                                        )

    # add gray outlines for nodes that survive the pthreshold
    x = pd.read_csv('/Users/taylorchamberlain/code/fc_ts_utils/fc_utils/network_definitions/xilin_268parc_labels.csv')
    sig_nodes_uncorr = [x for x in sig_nodes if x not in sig_nodes_corrected]
    display1.add_markers(
                marker_coords=x[x['Node_No'].isin(sig_nodes)][['MNI_X' ,'MNI_Y', 'MNI_Z']].values, 
                marker_size=60,
                marker_color=['dimgray'],
                alpha=.6,
                     marker= "o",
            edgecolor='dimgray',
    )

        
    display1.add_markers(
        marker_coords=x[x['Node_No'].isin(sig_nodes_corrected)][['MNI_X' ,'MNI_Y', 'MNI_Z']].values, 
        marker_size=110,
                    marker_color=['black'],
                                edgecolor='black',

                )



def create_figure_for_dataset(ds, df,   display_mode='yx', run='movieDM', vmax=.25, tasks=[], cmap='Spectral_r', use_controls=True):
    n_tasks = len(tasks)
    functions = ['annak', 'nn']
    fig, axes =  plt.subplots(n_tasks, 4, figsize=(25,n_tasks*4), sharex=False, sharey=False, gridspec_kw={'width_ratios':[2,1, 2, 1],})
    plt.subplots_adjust(wspace=.18, hspace=.16)
    for task_i, task in enumerate(tasks):
        for func in functions:
            func_i = 0
            if func=='annak':
                func_i = 2

            # plot glass brain
            ax_brain = axes[task_i][func_i]
            df=df[df['run']==run]
            sig_nodes, sig_nodes_corrected, r_vals = compare_outputs(ds, func, task, df, use_controls=use_controls)
            if r_vals.shape[0]>0:
                make_glass_brain_dots(range(268),
                                sig_nodes,
                                r_vals,
                                display_mode=display_mode,
                                sig_nodes_corrected=sig_nodes_corrected,
                                cmap=cmap, 
                                vmax=vmax,
                            ax=ax_brain)

            if 2 in df['half'].values:
                # plot split half scatter plot
                ax_scatter = axes[task_i][func_i+1]
                r1, r2, hue = compare_outputs_split_half(ds, func, task, df, run=run)
                x_ticks = False
                vmax_scatter=.2
                make_scatterplot(r1, r2, hue, ax_scatter, vmax=vmax_scatter, x_ticks=x_ticks)
    return fig


