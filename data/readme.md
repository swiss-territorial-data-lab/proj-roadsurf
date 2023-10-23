# Data
The data corresponds to the year 2018. <br>
All the initial georeferenced data is available in the crs EPSG:2056. <br>
All the initial georeferenced objects are represented as polygons, except for the roads which are represented as lines. <br>

## Area of interest
The area of interest (AOI) was defined with the beneficiaries of this project. It is constituted of 4 tiles from the 1:25'000 Swiss national map. This region contains a large and representative panel of topographies from the Swiss landscape. <br>
Having a restricted AOI of two tiles allows us to train the model on a zone and to test it on a new one.

## Quarries
The quarries were defined in [our project for quarries detection](https://github.com/swiss-territorial-data-lab/proj-dqry). However, it is also possible to deduce them from the product swissTLM3D by extracting the quarrying area for gravel, clay and stone from the feature class on area of use.

## swissTLM3D
The product [swissTLM3D](https://www.swisstopo.admin.ch/en/geodata/landscape/tlm3d.html) is the topographic model of Switzerland. It contains all the elements of the national map in the form of vector. <br>
The roads were taken from the feature class *TLM_STRASSEN*. The forests were extracted from the feature class *TLM_BODENBEDECKUNG* by selecting the cover types "forest" and "open forest".

## Road parameters
The parameters for the road treatment were defined depending on the class(es) of interest and by iterations for the road width. They are used to transform the roads from lines to polygons.

## Images
No images are provided in this repository. The images swissIMAGE 10 cm are available at the link given in the config files. The images swissIMAGE RS can be ordered from swisstopo and need then to go through the preprocessing phase. Both product are described on [the swisstopo dedicated webpage](https://www.swisstopo.admin.ch/en/geodata/images/ortho.html).


