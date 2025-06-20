# paper_CLMT

This is the PyTorch implementation for paper "Multi-Modal Contrastive Learning Based on Molecular and Textual data for Drug Response Prediction".


## Environment:
The codes of CLMT are implemented and tested under the following development environment:
    - Python 3.8
    - pandas==2.0.3
    - numpy==1.24.4
    - torch==1.10.0+cu113
    - torch-cluster==1.6.0
    - torch-geometric==2.1.0
    - torch-scatter==2.0.9
    - torch-sparse==0.6.13
    - torch-spline-conv==1.2.1

## Datasets

The files are too large to upload, but Google Drive download links are available in both the GDSC and CCLE folders.

Omics Datasets utilized in this paper are consistent with the previous work [A Survey and Systematic Assessment of Computational Methods for Drug Response Prediction](https://github.com/Jinyu2019/Suppl-data-BBpaper).

./CCLE_data/cell_descriptions.json and ./CCLE_data/drug_descriptions.csv are the textual description of the CCLE collected in this article.

./GDSC_data/cell_descriptions.json and ./CCLE_data/drug_descriptions.csv are the textual description of the GDSC collected in this article.

## Code Files:
The introduction of each <code> py </code> file is as follows:


- <i>prepare_data</i>: Code for data preprocessing.
- <i>train_model</i>: Code for training and testing the model function.
- <i>NeuML folder</i>: Download files from HuggingFace (https://huggingface.co/NeuML/pubmedbert-base-embeddings/tree/main).
- <i>model folder</i>: code for drug model and CLMT model.
- <i>train_CLMT.py.py</i>: The implementation of model training.
- <i>parse_args.py</i>: The parameter settings.


# How to use:
## 1. Generate dataset:
    Run create_cell_feat.py to generate the used cell_line feature.
    Run create_drug_feat.py to generate the used drug feature. 
    Run create_drp_dict.py to generate the drug_cell_ic50 feature and train_test split for mix,cell_blind and drug_blind sets.
    All the used data are saved in .npy format and under the ./GDSC/DRP_dataset or ./CCLE/DRP_dataset
    
## 2. Train the model:
    python train_CLMT.py


# Acknowledgments
This project is built with the reference to the following open-source projects:
[HiGeoMVP](https://github.com/matcyr/Hi-GeoMVP).