# High performers demonstrate greater neural synchrony than low performers across behavioral domains 

This repo contains analysis code for [Chamberlain et al., 2024](https://doi.org/10.1162/imag_a_00128).
The code was adapated from Emily Finn's code here: https://github.com/esfinn/intersubj_rsa. See the associated paper [here](https://www.sciencedirect.com/science/article/pii/S1053811920303153)
  

# Data

We analyzed two datasets, Healthy Brain Network and Yale Attention. The Yale Attention dataset can be downloaded [here](https://nda.nih.gov/edit_collection.html?id=2402).

The Healthy Brain Network imaging data are freely available and the timeseries data used for the paper's analysis are included in this repo. To download the raw Healthy Brain Network data see: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_EEG.html. The Healthy Brain Network behavioral data, however, require a Data Usage Agreement (see https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Phenotypic.html for details).

To make the code run smoothly as-is, I've included fake behavioral data in `hbn_behav.csv`. So please note that to get real results, you need to first submit a  DUA and download the behavioral data!


# Running the code
First install the necessary packages  `conda env create environment.yml`

Then run the code in the notebook `run_isrsa.ipynb`