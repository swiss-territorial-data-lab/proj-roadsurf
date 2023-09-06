
# Determination of the type of road surface

### Table of content

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting started](#getting-started)
    - [Folder structure](#folder-structure)
    - [Workflow](#workflow)
- [Other Uses](#other-uses)
    - [Preprocessing](#preprocessing)
    - [Machine-learning procedure](#machine-learning-procedure)


## Introduction

The aim of this project is to classify the roads of Switzerland according to whether they have an artificial or natural surface. The final objective was to integrate this information into [swissTLM3D](https://www.swisstopo.admin.ch/fr/geodata/landscape/tlm3d.html), the 3D topographic model of Switzerland. <br>
Using a F1 score with the same importance to both classes (artificial and natural), the final F1 score is 0.737 for the training, validation and test area and 0.557 for the inference-only area. <br>

The detailed documentation of the project can be found on the [STDL technical website](https://tech.stdl.ch/).

The input data are described in the dedicated `data` folder.


## Installation
The procedure was performed on Ubuntu 20.04. <br>

In order to run the project, the following elements are needed:
- this repository,
- the [object detector](https://github.com/swiss-territorial-data-lab/object-detector) repository,
- a CUDA-capable system.

To prepare the environment:

1. create a Python 3.8 virtual environment
2. if GDAL is not installed yet, run the following command:
```bash 
sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
```
3. intall the dependencies:
```bash
pip install -r requirements.txt
```

## Getting started

### Folder structure
```
.
├── config                      # Configuration files for the scripts and detectron2
├── data                        # Input data
├── img                         # Image folder of the readme
├── scripts
|   ├── functions               # Functions files
|   ├── preprocessing           # One-time scripts used in preprocessing
|   ├── road_segmentation       # Scripts used in the procedure based on the road segmentation
|   ├── sandbox                 # Scripts that were not implemented in the final procedure
|   ├── statistical_analysis    # Scripts used in the procedure based on the supervised classification
```

### Workflow

### Deep-learning

Road classification is performed with object segmentation and classification method. The scripts can be configured through the file `config_od.yaml`. <br>

<figure align="center">
<image src="img/road_segmentation_flow.jpeg" alt="flow for the road segmentation">
</figure>

The deep-learning procedure can be run with the following commands:
```bash
python scripts/road_segmentation/prepare_data_od.py config/config_od.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_od.yaml
python <path to the object detector>/scripts/train_model.py config/config_od.yaml
python <path to the object detector>/scripts/make_predictions.py config/config_od.yaml
python scripts/road_segmentation/final_metrics.py
```

### Machine-learning

Supervised classification of roads cover is performed with statistical approaches. The scripts can be configured through the file `config_stats.yaml`. <br>

<figure align="center">
<image src="img/statistical_flow.jpeg" alt="flow for the research of a statistical differences">
</figure>


The machine-learning method can be run with the following commands:
```bash
python scripts/statistical_analysis/prepare_data.py config/config_stats.yaml
python scripts/road_segmentation/prepare_data_od.py config/config_stats.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_stats.yaml
python scripts/statistical_analysis/statistical_analysis.py config/config_stats.yaml
```

## Other uses

### Preprocessing
Here, the included WMTS link points to the [SWISSIMAGE 10 cm](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html) product. Better results are achieved when using the [SWISSIMAGE RS](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage-rs.html) product and processing it to a WMTS-type service as described in the documentation. <br>
Produced images can be transfered from a local hard disk to our S3 cloud using the script `RS_images_to_S3.py`.
The script `tif2cog.py`, convert the images from 16-bit TIFF to 8-bit Cloud Optimized GeoTiff files and TiTiler was used to access them as tiles in a WMTS service.

```
python scripts/preprocessing/RS_images_to_S3 config/config_preprocessing.yaml
python scripts/preprocessing/tif2cog.py config/config_preprocessing.yaml
```