
# Determination of the type of road surface

### Table of content

- [Description](#description)
    - [Goal](#goal)
    - [Data](#data)
    - [Method](#method)
    - [Results](#results)
- [Installation](#installation)
- [Getting started](#getting-started)
    - [Folder structure](#folder-structure)
    - [Workflow](#workflow)
- [Other Uses](#other-uses)
    - [Preprocessing](#preprocessing)
    - [Statistical procedure](#statistical-procedure)


## Description

### Goal
The goal of this project is to classify the roads of Switzerland based on the type of their surface, artificial or natural. This work is currently done by operators at the Swiss Federal Office of Topography (swisstopo) and this is a time-consuming and repetitive task, therefore adapted to the automatization thank to data science. <br>

### Data
Initial data:
- [swissTLM3D](https://www.swisstopo.admin.ch/en/geodata/landscape/tlm3d.html):
    - roads as line,
    - forests as polygons,
- [SWISSIMAGE](https://www.swisstopo.admin.ch/en/geodata/images/ortho.html):
    - SWISSIMAGE 10 cm,
- Area of interest (AOI):
    - 4 tiles of the 1:25'000 national map situated in the region of the Emmental,
- quarries as polygons.

The starting point of the method are the line of the roads from the product swissTLM3D. Only the class "3m Strasse" was a problem for the operators, so the procedure is focused on this particular class. <br>
Here, the images used are the ones from the product SWISSIMAGE 10 cm made available by swisstopo in a WMTS service. Better results are achieved when using the product SWISSIMAGE RS and processing it to a WMTS service like proposed in the part [Other uses](#other-uses). However, this procedure is more complicated and time-consuming. <br>

### Method

<figure align="center">
<image src="img/proj_roadsurf_flow.jpeg" alt="Diagram of the methodology" style="width:60%;">
<figcaption align="center">Simplified diagram of the methodology for this project.</figcaption> 
</figure>

We first searched for statistical differences between the classes in order to perform a supervised classification. As we could not find any significative difference between the roads, we used artificial intelligence for the detection and classification of the roads. The [object detector of the STDL](https://github.com/swiss-territorial-data-lab/object-detector) was used. It is based on [detectron2 by FAIR](https://github.com/facebookresearch/detectron2). <br>
The procedure based on road segmentation is presented in priority here. The statistical procedure can be found under the section [Other uses](#other-uses).

### Results
Table of the f1-scores for each class and for the global results: 
|           	| Training, validation and test area 	| Other area 	|
|:---          	|:---:                                  |:---: 	        |
| Artificial 	|               0.959     	            |     0.916    	|
| Natural    	|               0.616     	            |     0.134    	|
| Global  	    |               0.790     	            |     0.547    	|


When using a f1-score giving the same importance to the two classes (artificial and natural), the final f1-score is 0.737 over the training, validation and test area and 0.557 over the other area. <br>

The detailed documentation can be found on [the technical website of the STDL](https://tech.stdl.ch/).

## Installation
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
To reproduce exactly the procedure described in the technical documentation, you will have to use the product SWISSIMAGE RS instead of SWISSIMAGE 10 cm. We obtained it on a hard disk and transferred it to our S3 cloud with the script `RS_images_to_S3.py`.
Then, with the help of the script `tif2cog.py`, the images were transformed from 16-bits TIFF to 8-bits Cloud Optimized GeoTiff files and Titiler was used to access them like tiles in a WMTS service.

```
python scripts/preprocessing/RS_images_to_S3 config/config_preprocessing.yaml
python scripts/preprocessing/tif2cog.py config/config_preprocessing.yaml
```

### Statistical procedure

<figure align="center">
<image src="img/statistical_flow.jpeg" alt="flow for the research of a statistical differences">
</figure>

Supervised classification was tested before road segmentation and classification. However, it was given up as we could not find significant statistical differences between the classes. The procedure was the following:

```
python scripts/statistical_analysis/prepare_data.py config/config_stats.yaml
python scripts/road_segmentation/prepare_data_od.py config/config_stats.yaml
python <path to the object detector>/scripts/generate_tilesets.py config/config_stats.yaml
python scripts/statistical_analysis/statistical_analysis.py config/config_stats.yaml
```
