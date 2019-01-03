
# coding: utf-8

# # Apply Machine Learning Classifiers to PDX RNAseq data
# 
# **Gregory Way, 2018**
# 
# In the following notebook, I apply three distinct classifiers to patient derived xenograft (PDX) RNAseq data (FPKM).
# The first classifier detects Ras activation. For more details about the algorithm and results, refer to [Way et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.046 "Machine Learning Detects Pan-cancer Ras Pathway Activation in The Cancer Genome Atlas"). I also include _TP53_ inactivation predictions. This classifier was previously applied in [Knijnenburg et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.076 "Genomic and Molecular Landscape of DNA Damage Repair Deficiency across The Cancer Genome Atlas"). The third is a classifier that predicts _NF1_ inactivation. We previously applied this in [Way et al. 2017](https://doi.org/10.1186/s12864-017-3519-7 "A machine learning classifier trained on cancer transcriptomes detects NF1 inactivation signal in glioblastoma").
# 
# To apply other classifiers (targeting other genes of interest) refer to https://github.com/greenelab/pancancer.
# 
# Also note that we have implemented classifiers on a larger scale and made the models accessible to non-computational biologists (see [Project Cognoma](http://cognoma.org)).
# 
# ## Procedure
# 
# 1. Load RNAseq matrix (`.RDS`)
#   * The matrix is in `sample` x `gene symbol` format (250 x 51,968)
# 2. Process matrix
#   * Take the z-score by gene
# 3. Load classifier coefficients
#   * For both _Ras_, _TP53_, and _NF1_ classifiers
# 4. Apply each classifier to the input data
#   * This also requires additional processing steps to the input data (subsetting to the respective classifier genes)
#   * Also note that not all genes are present in the input RNAseq genes.
# 5. Shuffle the gene expression data (by gene) and apply each classifier to random data
# 
# ### Important Caveat
# 
# Because not all of the classifier genes are found in the input dataset, the classifier probability is not calibrated correctly. The scores should be interpreted as continuous values representing relative gene alterations, and not as a pure probability estimate.
# 
# ## Output
# 
# The output of this notebook are the predicted scores for both classifiers across all samples for real data and shuffled data. This is in the form of a single text file with three columns (`sample_id`, `ras_score`, `tp53_score`, `nf1_score`,  `ras_shuffle`, `tp53_shuffle`, `nf1_shuffle`).

# In[1]:


import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from utils import apply_classifier, shuffle_columns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


np.random.seed(123)


# In[4]:


pandas2ri.activate()
readRDS = robjects.r['readRDS']


# In[5]:


# Make results directory
os.makedirs('./results', exist_ok=True)


# In[6]:


# Load PDX gene expression data in RDS format
file = os.path.join('data', 'raw', '2018-12-27-PPTC_FPKM_matrix_withModelID-248.RDS')

exprs_rds = readRDS(file)
exprs_df = pandas2ri.ri2py(exprs_rds).set_index('gene_short_name').transpose()

print(exprs_df.shape)
exprs_df.head(3)


# In[7]:


# Transform the gene expression data (z-score by gene)
scaled_fit = StandardScaler().fit(exprs_df)
exprs_scaled_df = pd.DataFrame(scaled_fit.transform(exprs_df),
                               index=exprs_df.index,
                               columns=exprs_df.columns)
exprs_scaled_df.head()


# In[8]:


# Shuffle input RNAseq matrix and apply classifiers
exprs_shuffled_df = exprs_scaled_df.apply(shuffle_columns, axis=0)


# ## Apply Ras Classifier

# In[9]:


# Load RAS Classifier
file = os.path.join('data', 'ras_classifier_coefficients.tsv')
ras_coef_df = pd.read_table(file, index_col=0)
ras_coef_df = ras_coef_df.query('abs_val > 0')

print(ras_coef_df.shape)
ras_coef_df.head()


# In[10]:


# Apply the Ras classifier to the input RNAseq matrix
ras_scores_df, ras_common_genes_df, ras_missing_genes_df = (
    apply_classifier(coef_df=ras_coef_df, rnaseq_df=exprs_scaled_df)
)


# In[11]:


# Determine the extent of coefficient overlap
print('There are a total of {} out of {} genes in common ({}%) between the datasets'
      .format(ras_common_genes_df.shape[0],
              ras_coef_df.shape[0],
              round(ras_common_genes_df.shape[0] / ras_coef_df.shape[0] * 100, 2)))


# In[12]:


# Which Genes are Missing?
ras_missing_genes_df


# In[13]:


# Distribution of predictions of the Ras Classifier applied to input data
ras_scores_df.T.hist(bins=30);


# ### Apply Ras Classifier to Shuffled Data

# In[14]:


# Apply the Ras classifier to the input RNAseq matrix 
ras_shuffle_scores_df, ras_shuffle_common_genes_df, ras_shuffle_missing_genes_df = (
    apply_classifier(coef_df=ras_coef_df, rnaseq_df=exprs_shuffled_df)
)


# ## Apply TP53 Classifier

# In[15]:


# Load RAS Classifier
file = os.path.join('data', 'tp53_classifier_coefficients.tsv')
tp53_coef_df = pd.read_table(file, index_col=0)
tp53_coef_df = tp53_coef_df.query('abs_val > 0')

print(tp53_coef_df.shape)
tp53_coef_df.head()


# In[16]:


# Apply the TP53 classifier to the input RNAseq matrix
tp53_scores_df, tp53_common_genes_df, tp53_missing_genes_df = (
    apply_classifier(coef_df=tp53_coef_df, rnaseq_df=exprs_scaled_df)
)


# In[17]:


# Determine the extent of coefficient overlap
print('There are a total of {} out of {} genes in common ({}%) between the datasets'
      .format(tp53_common_genes_df.shape[0],
              tp53_coef_df.shape[0],
              round(tp53_common_genes_df.shape[0] / tp53_coef_df.shape[0] * 100, 2)))


# In[18]:


# Which Genes are Missing?
tp53_missing_genes_df


# In[19]:


# Distribution of predictions of the TP53 Classifier applied to input data
tp53_scores_df.T.hist(bins=30);


# ### Apply TP53 Classifier to Shuffled Data

# In[20]:


# Apply the Ras classifier to the input RNAseq matrix 
tp53_shuffle_scores_df, tp53_shuffle_common_genes_df, tp53_shuffle_missing_genes_df = (
    apply_classifier(coef_df=tp53_coef_df, rnaseq_df=exprs_shuffled_df)
)


# ## Apply NF1 Classifier

# In[21]:


# Load NF1 Classifier
file = os.path.join('data', 'nf1_classifier_coefficients.tsv')
nf1_coef_df = pd.read_table(file, index_col=0)
nf1_coef_df = nf1_coef_df.query('abs_val > 0')

print(nf1_coef_df.shape)
nf1_coef_df.head()


# In[22]:


# Apply the NF1 classifier to the input RNAseq matrix
nf1_scores_df, nf1_common_genes_df, nf1_missing_genes_df = (
    apply_classifier(coef_df=nf1_coef_df, rnaseq_df=exprs_scaled_df)
)


# In[23]:


# Determine the extent of coefficient overlap
print('There are a total of {} out of {} genes in common ({}%) between the datasets'
      .format(nf1_common_genes_df.shape[0],
              nf1_coef_df.shape[0],
              round(nf1_common_genes_df.shape[0] / nf1_coef_df.shape[0] * 100, 2)))


# In[24]:


# Which Genes are Missing?
nf1_missing_genes_df


# In[25]:


# Distribution of predictions of the NF1 Classifier applied to input data
nf1_scores_df.T.hist(bins=30);


# ### Apply NF1 Classifier to Shuffled Data

# In[26]:


# Apply the NF1 classifier to the input RNAseq matrix 
nf1_shuffle_scores_df, nf1_shuffle_common_genes_df, nf1_shuffle_missing_genes_df = (
    apply_classifier(coef_df=nf1_coef_df, rnaseq_df=exprs_shuffled_df)
)


# ## Combine Ras and TP53 predictions and output to file

# In[27]:


results_list = [ras_scores_df.T, tp53_scores_df.T, nf1_scores_df.T,
                ras_shuffle_scores_df.T, tp53_shuffle_scores_df.T,
                nf1_shuffle_scores_df.T]

all_results = pd.concat(results_list, axis='columns').reset_index()
all_results.columns = ['sample_id', 'ras_score', 'tp53_score', 'nf1_score',
                       'ras_shuffle', 'tp53_shuffle', 'nf1_shuffle']

file = os.path.join('results', 'classifier_scores.tsv')
all_results.to_csv(file, sep='\t', index=False)

all_results.head()


# In[28]:


all_results.plot(kind='scatter', x='ras_score', y='nf1_score');

