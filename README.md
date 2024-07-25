# TMDs-multimaterials
We present deep learning models trained on AFM images of five classes of metal-organic
chemical deposition (MOCVD) grown TMDs. A systemic evaluation of transfer learning
strategies to accommodate low-data scenarios in materials synthesis and a model la-
tent feature analysis to draw connections to the human-interpretable characteristics of
the samples are presented.

# Objective
- to evaluate different modalities of transfer learning, including sequential learning, on classes of TMDs.
- to determine how best to adopt transfer learning to accommodate multiple classes of materials, includind low-data scenarios.
- to interprete the trained computer vision models by determining the physically meaningful properties of the AFM images that correlate with the latent features.

# Data
Processed data for this project can be found at 
Raw data can be found at https://data.2dccmip.org/Rut1mMC8u25M

# Workflows
- Data: The features extracted from the images using untrained ResNet18 and trained ResNet152 models. The former features are used in training MLP model. The main processed data is available in the zenodo.
- Train: Contains the Python and bash scripts used in training the models. Also contains the MLP notebook.
- codes: codes
- Models: The trained models
- Analysis_and_Plots: Notebooks used for analysing the results and for generating plots
- Results: Some result files and plots
. 
# To cite
@article{moses2024crystal,
  title={Transfer Learning for Multi-material Classification of Transition Metal Dichalcogenides with Atomic Force Microscopy},
  author={Moses, Isaiah A and Reinhart, Wesley F},
  journal={},
  volume={},
  pages={},
  year={2024},
  publisher={}
}
