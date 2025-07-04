# Does this smell the same? Learning representations of olfactory mixtures using inductive biases

![image](figures/figure1.png)

## Getting started

Create a conda environment:
```bash
conda env create -f environment.yml
conda activate pommix
```


## Datasets
The `datasets` folder contains the olfactory properties of single molecules, obtained from the [OpenPOM version](https://github.com/BioMachineLearning/openpom) of the GoodScents-Leffingwell (GS-LF) dataset. This folder also contains the olfactory similarity metrics for mixtures, contained in `mixtures_combined.csv` obtained from [Snitz et al. (2013)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003184), [Bushdid et al. (2014)](https://www.science.org/doi/10.1126/science.1249168) and [Ravia et al. (2020)](https://www.nature.com/articles/s41586-020-2891-7). The SMILES strings of the components in the multimolecular mixtures can be found in `mixture_smi_definitions_clean_found_duplicates.csv`. 

## Splits

The `datasets` folder also contains the pre-determined random splits (prefix `random`), cross-validation splits (prefix `random_cv`), splitting by number of components in a mixture (in Figure 5a, prefix `ablate_components`), and the leave-molecules-out splits (LMO, in Figure 5b, prefix `lso_molecules`)

## Baselines (Figure 4)

The `datasets` folder contains the features used for the RDKit baselines `mixture_rdkit_definitions_clean_found_duplicates.csv` and the principal odor map (POM) embeddings used for the XGBoost models `mixture_pom_embeddings.npz`. To run the baselines, use the scripts and splits contained in the `scripts/baseline` folder.

## Models

![image](figures/figure3.png)

### Pretraining POM

`scripts/pom` contains the code to generate olfactory molecular embeddings by predicting the odor descriptors for a molecule based on the GS-LF dataset. 

```bash
# run pretraining
# parameters determined from hyperparameter optimization
# model saved in `gs-lf_models/pretrained_pom`
python run_pretraining.py --depth=4 --hidden_dim=320 --dropout=0.1 --lr=0.0001 --tag='pretrained_pom'
```

### Pretraining CheMix

`scripts/chemix` contains the code to take mixture embeddings to predict the olfactory similarity of two mixtures. Pretrained POM is pulled from the previous step.

```bash
# run chemix with pom embeddings
# parameters are from default.yaml, which are determined from hyperparameter optimization
# model saved in `results/{split_type}/`
python run_chemix.py --split={split_type}
```

`scripts/chemix/data_utilities` contain scripts used to generate the mixture features and splits.


### Train/fine-tune POMMix

`scripts/pommix` contains the code that trains both POM and CheMix end-to-end to improve the predictive performance of molecular embeddings generated from the POM. The POM model pulled from first step. The CheMix model is pulled from the same splits used as previous step.

```bash
# run POMMix
# parameters generated from previous steps
# model saved in `results/{split_type}/`
python run_pommix.py --split={split_type}
```

### Baselines

`scripts/baseline` contains baseline models used for comparison. This includes Snitz baseline, and XGBoost with RDKit features.

To perform Snitz baseline, the three steps as detailed in the appendix are separated into 3 scripts:
```bash
python run_snitz_step1.py
python run_snitz_step2.py
python run_snitz_step3.py     # final features are saved in `snitz_step3`

python run_snitz.py           # results saved in `snitz_similarity`
```

To perform XGBoost models for either RDKit or POM features:
```bash
python run_xgb.py --feat_type={"rdkit2d", "pom_embeddings", "molt5embeddings"}     # results saved in `xgb_{feat_type}`. Molt5 will need to be generated
```

## Figures

All figures generated in the paper can be found using scripts found in the `figures` folder -- most of them are in `figures.ipynb`. 


## Additional ablation experiments

### Feature ablation

To further study the performance of the POM embeddings, we also test the use of MolT5 ([Edwards et al. 2022](https://arxiv.org/pdf/2204.11817)) embeddings. The embeddings can be generated using the [molfeat](https://molfeat.datamol.io/) package. The script to generate the embeddings are created in the `scripts/chemix/data_utilities/make_molt5.py`.

We use the embeddings with XGB in `scripts/baseline/` and also CheMix `scripts/chemix/`.


### GNN ablation

To test the GNN archiecture, we also perform experiments with the [Graphormer](https://github.com/microsoft/Graphormer) and the [GPS](https://arxiv.org/abs/2205.12454) models, which have performed SOTA on larger datasets.

We direct the user to use the repository for Graphromer, we use the default of Graphromer-slim. For GPS, the script is found in `scripts/pom/gpsconv.py`.


### Prediction head ablation

The model of CheMix has been modified with different prediction heads. The results are found in `scripts/chemix/results/random_cv/`. We finally use the scaled cosine distance, which is reflected in the final model architecture.

