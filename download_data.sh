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
wget --output-document='data/raw/2019-02-14-PPTC_FPKM_matrix_withModelID-244.rda' \
  https://ndownloader.figshare.com/files/14452985

####################################
# Genomic alterations data
####################################
wget --output-document='data/raw/2019-02-14-ras-tp53-nf1-alterations.txt' \
  https://ndownloader.figshare.com/files/14372792

####################################
# Clinical data
####################################
wget --output-document='data/raw/pptc-pdx-clinical-web.txt' \
  https://ndownloader.figshare.com/files/14508536

####################################
# Checksums
####################################
# Confirm the integrity of downloaded data
md5sum -c data/md5sums.txt
