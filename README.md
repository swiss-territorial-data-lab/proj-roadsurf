
# Determination of the type of road surface

### Table of content

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting started](#getting-started)
    - [Folder structure](#folder-structure)
    - [Deep learning workflow](#deep-learning-workflow)
- [Additional uses](#additional-uses)
    - [Preprocessing](#preprocessing)
    - [Machine-learning workflow](#machine-learning-workflow)


## Introduction

The aim of this project is to classify the roads of Switzerland according to whether they have an artificial or natural surface. The final objective was to integrate this information into [swissTLM3D](https://www.swisstopo.admin.ch/fr/geodata/landscape/tlm3d.html), the 3D topographic model of Switzerland. <br>
Using a F1 score with the same importance to both classes (artificial and natural), the final F1 score is 0.775 for the validation area and 0.548 for the inference-only area. The algorithm has an approach based on deep learning using the STDL's object detector.<br>
A procedure based on machine learning was tested, but not completed as no significant statistical difference could be found between the classes.

The detailed documentation can be found on the [STDL technical website](https://tech.stdl.ch/PROJ-ROADSURF/).

The input data are described in the dedicated `data` folder.


## Installation
The procedure was performed on Ubuntu 20.04. <br>

The following elements are needed:
- this repository,
- the [object detector repository](https://github.com/swiss-territorial-data-lab/object-detector),
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
|   ├── preprocessing           # Scripts used in preprocessing
|   ├── road_segmentation       # Scripts used in the procedure based on deep learning and using the STDL's object detector
|   ├── sandbox                 # Scripts that were not implemented in the final procedure
|   ├── statistical_analysis    # Scripts used in the procedure based on machine learning
```

### Deep learning workflow

The road, initially provided as vector lines, were transformed to polygon labels to approximate the real extent of roads in the images. Then, a model was trained to classify roads in 2 classes (natural or artificial) with the STDL's object detector. Finally, the results were assessed. The entire procedure is illustrated in the diagram below.

<figure align="center">
<image src="img/road_segmentation_flow.jpeg" alt="flow for the road segmentation">
</figure>

The scripts can be configured through the file `config_obj_detec.yaml` and `detectron2_config_3bands.yaml`. <br>

The deep learning algorithm can be run with the following commands:
```bash
python scripts/road_segmentation/prepare_data_obj_detec.py config/config_obj_detec.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_obj_detec.yaml
python <path to the object detector>/scripts/train_model.py config/config_obj_detec.yaml
python <path to the object detector>/scripts/make_detections.py config/config_obj_detec.yaml
python scripts/road_segmentation/final_metrics.py
```

## Additional applications

### Image processing
The WTMS link in the file `config_obj_detec.yaml` refers to the [SWISSIMAGE 10 cm product](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html). However, better results are achieved using the [SWISSIMAGE RS product](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage-rs.html). These data are available on request from swisstopo and require specific data processings and an infrastructure to obtain tiles managed by a WMTS-like service as described in the documentation.<br>
The images were:
- transferred on a S3 cloud with the script `RS_images_to_S3.py`,
- transformed from 16-bit TIFF to 8-bit Cloud Optimized GeoTiff files with the script `tif2cog.py`.
TiTiler was used to access them as tiles in a WMTS service.

The scripts can be configured through the file `config_preprocessing.yaml`. <br>

```
python scripts/preprocessing/RS_images_to_S3.py config/config_preprocessing.yaml
python scripts/preprocessing/tif2cog.py config/config_preprocessing.yaml
```

### Machine-learning workflow

Supervised classification was tested before road segmentation and classification. However, it was given up as we could not find significant statistical differences between the classes. The procedure is described here below.

<figure align="center">
<image src="img/statistical_flow.jpeg" alt="flow for the research of a statistical differences">
</figure>

The scripts can be configured through the file `config_stats.yaml`. <br>

```
python scripts/statistical_analysis/prepare_data.py config/config_stats.yaml
python scripts/road_segmentation/prepare_data_obj_detec.py config/config_stats.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_stats.yaml
python scripts/statistical_analysis/statistical_analysis.py config/config_stats.yaml
```
