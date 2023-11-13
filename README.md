# croptypemapper
This repo contains code for running temporal convolutional neural networks and Long Short-Term Memory Networks designed to map crop types at 10 m resolution using time series (typically 1 year) of Sentinel-1 and 2 data. This code was developed as part of the Enabling Crop Analytics at Scale project.  

The models is still being actively developed, as is the code-base. The model can currently be run using the code provided for both training and inference.  

A notebook demonstrating the use of a pre-trained model for predicting crop types within Google's colab environment is currently available [here](https://github.com/agroimpacts/croptypemapper/blob/main/croptypemapper/notebooks/predict_croptypes_colab_demo.ipynb). To run the notebook, several set of files are required, including:

- Sentinel-1 predictors for at least one tile
- Sentinel-2 predictors for the same tile(s)
- Pre-trained model parameter file 

An optional step is to filter the resulting crop type predictions using a geotiff of the same resolution and for the same tiles containing crop field classifications, so that model predictions are confined to crop fields.  

The introduction to the notebook describes these data in more detail.  




