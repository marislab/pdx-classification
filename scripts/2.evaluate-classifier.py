
# coding: utf-8

# # Evaluate Classifier Predictions
# 
# **Gregory Way, 2018**
# 
# In the following notebook I evaluate the predictions made by the Ras, _NF1_, and _TP53_ classifiers in the input PDX RNAseq data.
# 
# ## Procedure
# 
# 1. Load status matrices
#   * These files store the mutation status for _TP53_ and Ras pathway genes for the input samples
# 2. Align barcode identifiers
#   * The identifiers matching the RNAseq data to the status matrix are not aligned.
#   * I use an intermediate dictionary to map common identifiers
# 3. Load predictions (see `1.apply-classifier.ipynb` for more details)
# 4. Evaluate predictions
#   * I visualize the distribution of predictions between wild-type and mutant samples for both classifiers
# 
# ## Output
# 
# The output of this notebook are several evaluation figures demonstrating the predictive performance on the input data for the three classifiers. Included in this output are predictions stratified by histology.

# In[1]:


import os
import random
from decimal import Decimal
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve

import seaborn as sns
import matplotlib.pyplot as plt

from utils import get_mutant_boxplot, perform_ttest


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


np.random.seed(123)


# ## Load Status Matrix

# In[4]:


file = os.path.join('data', 'raw', '2019-01-14-ras-tp53-nf1-alterations.txt')
status_df = pd.read_table(file)

print(status_df.shape)
status_df.head(3)


# In[5]:


status_df.Hugo_Symbol.value_counts()


# In[6]:


status_df.Variant_Classification.value_counts()


# In[7]:


pd.crosstab(status_df['Histology.Detailed'], status_df.classifier)


# In[8]:


pd.crosstab(status_df['Histology.Detailed'], status_df.Hugo_Symbol)


# In[9]:


# Obtain a binary status matrix
full_status_df = pd.crosstab(status_df['Model'], status_df.Hugo_Symbol)
full_status_df[full_status_df > 1] = 1
full_status_df = full_status_df.reset_index()


# In[10]:


histology_df = status_df.loc[:, ['Model', 'Histology.Detailed']]
histology_df.columns = ['Model', 'Histology_Full']

full_status_df = (
    full_status_df
    .merge(histology_df, how='left', on="Model")
    .drop_duplicates()
    .reset_index(drop=True)
)

full_status_df.head()


# ## Extract Gene Status

# In[11]:


# Ras Pathway Alterations
ras_genes = ['KRAS', 'HRAS', 'NRAS']
tp53_genes = ["TP53", "RB1", "CHEK2", "MDM2", "MDM4"]

full_status_df = (
    full_status_df
    .assign(ras_status = full_status_df.loc[:, ras_genes].sum(axis=1),
            tp53_status = full_status_df.loc[:, tp53_genes].sum(axis=1),
            nf1_status = full_status_df['NF1'])
    
)

full_status_df.head()


# ## Load Clinical Data Information
# 
# This stores histology information

# In[12]:


file = os.path.join('data', 'raw', '2018-09-30-pdx-clinical-final-for-paper.txt')
clinical_df = pd.read_table(file)

print(clinical_df.shape)
clinical_df.head(3)


# ## Load Predictions and Merge with Clinical and Alteration Data

# In[13]:


file = os.path.join('results', 'classifier_scores.tsv')
scores_df = pd.read_table(file)

scores_df = (
    scores_df.merge(
        clinical_df,
        how='left', left_on='sample_id', right_on='Model'
    )
    .merge(
        full_status_df,
        how='left', left_on='sample_id', right_on='Model'
    )
)

print(scores_df.shape)
scores_df.head()


# In[14]:


gene_status = ['tp53_status', 'ras_status', 'nf1_status']
scores_df.loc[:, gene_status] = (
    scores_df.loc[:, gene_status].fillna(0)
)

scores_df.loc[scores_df['tp53_status'] != 0, 'tp53_status'] = 1
scores_df.loc[scores_df['ras_status'] != 0, 'ras_status'] = 1
scores_df.loc[scores_df['nf1_status'] != 0, 'nf1_status'] = 1

scores_df['tp53_status'] = scores_df['tp53_status'].astype(int)
scores_df['ras_status'] = scores_df['ras_status'].astype(int)
scores_df['nf1_status'] = scores_df['nf1_status'].astype(int)

scores_df.head(2)


# ## Load Histology Color Codes

# In[15]:


file = os.path.join('data', '2018-08-23-all-hist-colors.txt')
color_code_df = pd.read_table(file)
color_code_df.head(2)

color_dict = dict(zip(color_code_df.Histology, color_code_df.Color))
color_dict


# ## Determine Status Counts

# In[16]:


scores_df.tp53_status.value_counts()


# In[17]:


scores_df.ras_status.value_counts()


# In[18]:


scores_df.nf1_status.value_counts()


# ## Perform ROC and Precision-Recall Analysis using all Alteration Information

# In[19]:


n_classes = 3
labels = ['Ras', 'NF1', 'TP53']
colors = ['#1b9e77', '#d95f02', '#7570b3']

fpr_pdx = {}
tpr_pdx = {}
thresh_pdx = {}
precision_pdx = {}
recall_pdx = {}
auroc_pdx = {}
aupr_pdx = {}

fpr_shuff = {}
tpr_shuff = {}
thresh_shuff = {}
precision_shuff = {}
recall_shuff = {}
auroc_shuff = {}
aupr_shuff = {}

all_roc_list = []

idx = 0
for status, score, shuff in zip(('ras_status', 'nf1_status', 'tp53_status'),
                                ('ras_score', 'nf1_score', 'tp53_score'),
                                ('ras_shuffle', 'nf1_shuffle', 'tp53_shuffle')):
    
    # Obtain Metrics
    sample_status = scores_df.loc[:, status]
    sample_score = scores_df.loc[:, score]
    shuffle_score = scores_df.loc[:, shuff]
 
    # Get Metrics
    fpr_pdx[idx], tpr_pdx[idx], thresh_pdx[idx] = roc_curve(sample_status, sample_score, drop_intermediate=False)
    precision_pdx[idx], recall_pdx[idx], _ = precision_recall_curve(sample_status, sample_score)
    auroc_pdx[idx] = roc_auc_score(sample_status, sample_score)
    aupr_pdx[idx] = average_precision_score(sample_status, sample_score)
    
    # Obtain Shuffled Metrics
    fpr_shuff[idx], tpr_shuff[idx], thresh_shuff[idx] = roc_curve(sample_status, shuffle_score, drop_intermediate=False)
    precision_shuff[idx], recall_shuff[idx], _ = precision_recall_curve(sample_status, shuffle_score)
    auroc_shuff[idx] = roc_auc_score(sample_status, shuffle_score)
    aupr_shuff[idx] = average_precision_score(sample_status, shuffle_score)
    
    roc_df = (
        pd.DataFrame([fpr_pdx[idx], tpr_pdx[idx], thresh_pdx[idx]], index=['fpr', 'tpr', 'threshold'])
        .transpose()
        .assign(gene=labels[idx],
                shuffled=False)
    )
    
    roc_shuffled_df = (
    pd.DataFrame([fpr_shuff[idx], tpr_shuff[idx], thresh_shuff[idx]], index=['fpr', 'tpr', 'threshold'])
    .transpose()
    .assign(gene=labels[idx],
            shuffled=True)
    )
    
    all_roc_list.append(roc_df)
    all_roc_list.append(roc_shuffled_df)

    idx += 1


# In[20]:


os.makedirs('figures', exist_ok=True)


# In[21]:


# Visualize ROC curves
plt.subplots(figsize=(4, 4))

for i in range(n_classes):
    plt.plot(fpr_pdx[i], tpr_pdx[i],
             label='{} (AUROC = {})'.format(labels[i], round(auroc_pdx[i], 2)),
             linestyle='solid',
             color=colors[i])

    # Shuffled Data
    plt.plot(fpr_shuff[i], tpr_shuff[i],
             label='{} Shuffle (AUROC = {})'.format(labels[i], round(auroc_shuff[i], 2)),
             linestyle='dotted',
             color=colors[i])

plt.axis('equal')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)

plt.tick_params(labelsize=10)

lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=10)

file = os.path.join('figures', 'classifier_roc_curve.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')


# In[22]:


# Visualize PR curves
plt.subplots(figsize=(4, 4))

for i in range(n_classes):
    plt.plot(recall_pdx[i], precision_pdx[i],
             label='{} (AUPR = {})'.format(labels[i], round(aupr_pdx[i], 2)),
             linestyle='solid',
             color=colors[i])
    
    # Shuffled Data
    plt.plot(recall_shuff[i], precision_shuff[i],
             label='{} Shuffle (AUPR = {})'.format(labels[i], round(aupr_shuff[i], 2)),
             linestyle='dotted',
             color=colors[i])

plt.axis('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)

plt.tick_params(labelsize=10)

lgd = plt.legend(bbox_to_anchor=(1.03, 0.85),
                 loc=2,
                 borderaxespad=0.,
                 fontsize=10)

file = os.path.join('figures', 'classifier_precision_recall_curve.pdf')
plt.savefig(file, bbox_extra_artists=(lgd,), bbox_inches='tight')


# ## Perform t-test against status classification

# In[23]:


t_results_ras = perform_ttest(scores_df, gene='ras')
t_results_ras


# In[24]:


t_results_nf1 = perform_ttest(scores_df, gene='nf1')
t_results_nf1


# In[25]:


t_results_tp53 = perform_ttest(scores_df, gene='tp53')
t_results_tp53


# ## Observe broad differences across sample categories

# In[26]:


# Ras
get_mutant_boxplot(df=scores_df,
                   gene="Ras",
                   t_test_results=t_results_ras)


# In[27]:


# NF1
get_mutant_boxplot(df=scores_df,
                   gene="NF1",
                   t_test_results=t_results_nf1)


# In[28]:


# TP53
get_mutant_boxplot(df=scores_df,
                   gene="TP53",
                   t_test_results=t_results_tp53)


# In[29]:


# Ras Alterations
get_mutant_boxplot(df=scores_df,
                   gene='Ras',
                   histology=True,
                   hist_color_dict=color_dict)


# In[30]:


# NF1 Alterations
get_mutant_boxplot(df=scores_df,
                   gene='NF1',
                   histology=True,
                   hist_color_dict=color_dict)


# In[31]:


# TP53 Alterations
get_mutant_boxplot(df=scores_df,
                   gene='TP53',
                   histology=True,
                   hist_color_dict=color_dict)


# ## Write output files for downstream analysis

# In[32]:


# Classifier scores with clinical data and alteration status
scores_file = os.path.join("results", "classifier_scores_with_clinical_and_alterations.tsv")
genes = ras_genes + ['TP53']

scores_df = scores_df.drop(['Model_x', 'Model_y', 'Histology_Full'], axis='columns')
scores_df[genes] = scores_df[genes].fillna(value=0)

scores_df.sort_values(by='sample_id').to_csv(scores_file, sep='\t', index=False)


# In[33]:


# Output classifier scores for the specific variants observed
status_scores_file = os.path.join("results", "classifier_scores_with_variants.tsv")

classifier_scores_df = scores_df[['sample_id', 'ras_score' ,'tp53_score', 'nf1_score', 'Histology-Detailed']]
classifier_scores_df = (
    status_df
    .drop(['Histology.Detailed'], axis='columns')
    .merge(classifier_scores_df, how='left', left_on='Model', right_on='sample_id')
)

classifier_scores_df.sort_values(by='Model').to_csv(status_scores_file, sep='\t', index=False)


# In[34]:


# ROC Curve Estimates
file = os.path.join("results", "full_roc_threshold_results.tsv")

full_roc_df = pd.concat(all_roc_list, axis='rows')
full_roc_df.to_csv(file, sep='\t', index=False)

