
# Determination of the type of road surface

## Description

### Goal
The goal of this project is to classify the roads of Switzerland based on the type of their surface, natural or artificial. This work is currently done by operators of the Swiss Federal Office of Topography (swisstopo) and this is a time-consuming and repetitive task, therefore adapted to the automatization thank to data science. <br>

### Data
The starting point of the method are the line of the roads from the product swissTLM3D. Only the class "3m Strasse" was a problem for the operator, so the procedure is focused on this particular class. The images used are the ones from the product SWISSIMAGE RS.<br>

### Method
We first searched for statistical differences between the classes in order to perform a supervised classification. As we could not find any significative difference between the roads, we used an artificial intelligence for the instance segmentation for the detection and classification of the roads. The [object detector of the STDL](https://github.com/swiss-territorial-data-lab/object-detector) was used. It is based on [detectron2 by FAIR](https://github.com/facebookresearch/detectron2).

### Results
The final metrics were ...


The detailed documentation can be found on [the technical website of the STDL](https://tech.stdl.ch/).

## Installation
In order to run the project, you will need this repository as well as the one of the [object detector](https://github.com/swiss-territorial-data-lab/object-detector).<br>
A CUDA-capable system is required. <br>
To prepare the environment:

1. create a Python 3.8 environment
2. if you do not have GDAL installed, run the following command:
```bash 
sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
```
3. intall the dependencies:
```bash
pip install -r requirements.txt
```

## Getting started
All the scripts can be configured through the file `config.yaml`. <br>
In order to reproduce the results of the project, the initial data are available in the `script/data`.<br>
The method can be run with the following commands:
```bash
python scripts/road_segmentation/prepare_data_od.py config.yaml
python <path to the object detector>/scripts/generate_tilesets.py config.yaml
python <path to the object detector>/scripts/train_model.py config.yaml 
python <path to the object detector>/scripts/make_predictions.py config.yaml
python <path to the object detector>/scripts/assess_predictions.py config.yaml
python scripts/road_segmentation/final_metrics.py config.yaml
```