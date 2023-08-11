
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

In this project, the roads of Switzerland were classified according to whether they had an artificial or natural surface. The final objective was to integrate this information into the 3D topographic model of Switzerland. <br>
Using a F1 score giving the same importance to both classes (artificial and natural), the final F1 score is 0.737 for the training, validation and test area and 0.557 for the inference-only area. <br>

The full documentation can be found on the [STDL technical website](https://tech.stdl.ch/).

The initial data are described in the dedicated `data` folder.


## Installation
The procedure was tested on Ubuntu 20.04. <br>

In order to run the project, you will need :
- this repository,
- the repository of the [object detector](https://github.com/swiss-territorial-data-lab/object-detector),
- a CUDA-capable system.

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

### Folder structure
```
.
├── config                      # Configuration files for the scripts and detectron2
├── data                        # Initial data
├── img                         # Image folder for the readme
├── scripts
|   ├── functions               # Functions files
|   ├── preprocessing           # One-time scripts used in preprocessing
|   ├── road_segmentation       # Scripts used in the procedure based on the road segmentation
|   ├── sandbox                 # Scripts that were not implemented in the final procedure
|   ├── statistical_analysis    # Scripts used in the procedure based on the supervised classification
```

### Workflow
<figure align="center">
<image src="img/road_segmentation_flow.jpeg" alt="flow for the road segmentation">
</figure>

The scripts can be configured through the file `config_od.yaml`. <br>

The method can be run with the following commands:
```bash
python scripts/road_segmentation/prepare_data_od.py config/config_od.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_od.yaml
python <path to the object detector>/scripts/train_model.py config/config_od.yaml
python <path to the object detector>/scripts/make_predictions.py config/config_od.yaml
python scripts/road_segmentation/final_metrics.py
```

## Other uses

### Preprocessing
Here, the included WTMS link points the [SWISSIMAGE 10 cm](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html) product. Better results are achieved when using the [SWISSIMAGE RS](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage-rs.html) product and processing it to a WMTS-type service as described in the documentation. <br>
We obtained the images on a hard disk and transferred them to our S3 cloud using the script `RS_images_to_S3.py`.
Then, using the script `tif2cog.py`, the images were transformed from 16-bit TIFF to 8-bit Cloud Optimized GeoTiff files and TiTiler was used to access them as tiles in a WMTS service.

```
python scripts/preprocessing/RS_images_to_S3 config/config_preprocessing.yaml
python scripts/preprocessing/tif2cog.py config/config_preprocessing.yaml
```

### Machine-learning procedure

<figure align="center">
<image src="img/proj_roadsurf_flow.jpeg" alt="Diagram of the methodology" style="width:60%;">
<figcaption align="center">Simplified diagram of the methodology for this project.</figcaption> 
</figure>

Supervised classification was tested before road segmentation and classification. However, it was given up as we could not find significant statistical differences between the classes. The procedure is described here below.

<figure align="center">
<image src="img/statistical_flow.jpeg" alt="flow for the research of a statistical differences">
</figure>

```
python scripts/statistical_analysis/prepare_data.py config/config_stats.yaml
python scripts/road_segmentation/prepare_data_od.py config/config_stats.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_stats.yaml
python scripts/statistical_analysis/statistical_analysis.py config/config_stats.yaml
```
