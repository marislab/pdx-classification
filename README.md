# Apply Machine Learning Classifiers to Target Data

**Gregory Way, Jo Lynne Harenza, John Maris, 2018**

Here, we apply a Ras activation and TP53 inactivation classifier to Target Patient Derived Xenograft (PDX) RNAseq data.
The classifiers were previously trained using data from The Cancer Genome Atlas (TCGA) PanCanAtlas Project ([Way et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.046 "Machine Learning Detects Pan-cancer Ras Pathway Activation in The Cancer Genome Atlas"), [Knijnenburg et al. 2018](https://doi.org/10.1016/j.celrep.2018.03.076 "Genomic and Molecular Landscape of DNA Damage Repair Deficiency across The Cancer Genome Atlas")) 

## Computational Environment

We use [conda](https://conda.io/docs/user-guide/install/index.html) as an environment manager.
To reproduce the computational environment used in this pipeline, run:

```bash
# Using conda version >4.5
conda env create --force --file environment.yml

conda activate target-classification
```

## Pipeline

The following notebooks describe the analysis pipeline

| Notebook | Description |
| :------- | :---------- |
| `1.apply-classifier.ipynb` | Apply the classifiers trained previously on the input data |
| `2.evaluate-classifier.ipynb` | Investigate and evaluate the prediction performance and score distribution for input data |

To rerun all scripts, perform the following:

```bash
# Make sure to activate the conda environment
conda activate target-classification

# Run the pipeline to extract results, figures, and convert notebooks for easy viewing
./run_analysis.sh
```
