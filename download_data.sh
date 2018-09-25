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
wget --output-document='data/raw/2018-08-24-PPTC_FPKM_matrix_withModelID-250.RDS' \
  https://ndownloader.figshare.com/files/13113614


####################################
# Genomic alterations data
####################################
wget --output-document='data/raw/2018-09-21-muts-fusions.txt' \
  https://ndownloader.figshare.com/files/13113617

####################################
# Checksums
####################################
# Confirm the integrity of downloaded data
md5sum -c data/md5sums.txt
