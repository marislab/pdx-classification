#!/bin/bash

mkdir -p 'data/raw/'

####################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Download PPTC RNAseq and genomic alterations
# https://figshare.com/articles/PPTC_RNAseq_and_Genomic_Alterations/7127726
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
####################################

####################################
# RNAseq data
####################################
wget --output-document='data/raw/2018-12-27-PPTC_FPKM_matrix_withModelID-248.RDS' \
  https://ndownloader.figshare.com/files/14023604

####################################
# Genomic alterations data
####################################
wget --output-document='data/raw/2019-01-03-muts-fusions.txt' \
  https://ndownloader.figshare.com/files/14023742

####################################
# Checksums
####################################
# Confirm the integrity of downloaded data
md5sum -c data/md5sums.txt
