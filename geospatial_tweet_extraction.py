#!/usr/bin/env python
# coding: utf-8

# # Geospatial Tweet Extraction
# ---
# The following code sources geospatial administrative data, processes those data so that they may be used in the querying of the Twitter full archive search endpoint, perfroms that querying, and stores extracted tweet data locally. 

# ## Processing Geospatial Data
# ---
# The following code processes data which define administrative geographies sourced from the ONS Open Geography Portal (https://geoportal.statistics.gov.uk) to generate a dataset that may be used as an input to execute bounding box based queries of the Twitter API (https://developer.twitter.com/en/docs/twitter-api) for a specified Local Authority.

# ### Import Libraries
# ---

# In[ ]:


# Libraries required for processing of geo admin data
import geopandas as gpd
from geopandas import GeoSeries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import contextily as ctx
import openpyxl
import shapely
from shapely.geometry import Polygon, LineString, Point
import itertools

# Twarc python libraries, installation required
# twarc2 guidance: https://twarc-project.readthedocs.io/en/latest/twarc2_en_us/
from twarc import Twarc2, expansions
from twarc.client2 import Twarc2
from twarc_csv import CSVConverter, DataFrameConverter

# common Python libraries
import numpy as np     # https://numpy.org/
import pandas as pd    # https://pandas.pydata.org/

# in-built Python modules
import datetime     # https://docs.python.org/3/library/datetime.html#module-datetime
import os           # https://docs.python.org/3/library/os.html#module-os
import json         # https://docs.python.org/3/library/json.html#module-json
import csv          # https://docs.python.org/3/library/csv.html#module-csv
import random       # https://docs.python.org/3/library/random.html#module-random


# ### Set Global Variables
# ---
# 
# The variables below are intialised to reference the directory locations of the source data (a shape file containing vector boundaries for Output Areas, sourced from the ONS Open Geography Portal); the directory location to which outputs will be stored; the name of the Local Authority for which a bounding box grid is to be generated, and the directory locations of source bounding box data and the location of a .txt file containing the bearer token required to access the full-archive search end point of the official Twitter API V2.0 (https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all).
# 
# Variables are also initialise to specify the location of folders to which .jsonl files containing tweet data will be stored, and .csv files into which those .jsonl files will be converted are store. Each of these folders should contain a file for each year for which the extraction of data is to eb carried out. 
# 
# Finally, a list object is intialised containing all years for which tweet data are to be extracted, as intergers.

# In[ ]:


# intialise path variable specifying location of 
admin_geo_path = 'E:/twitter_data/june_22_extract/bbox_data/admin_geo_data_for _bb/'

# intialise path variable specifying desired location for storage of outputs
bb_ouput_path = 'E:/twitter_data/june_22_extract/bbox_data/bb_list_output/'

# initialise variable specifying the LA for which a bounding box grid will be generated
LA =  'Camden'

# initialise path variable citing location of bounding box values .csv file
bbox_data_path = 'E:/twitter_data/extracted_geocoded_tweets/bbox_inputs/'

# initialise path variable citing location of bearer token .txt file
bearer_token_path = 'E:/twitter_data/extracted_geocoded_tweets/bearer_token/'

# initialise path variable citing output location for .jsonl files extracted via Twitter API
# this folder is to contain sub folders labelled by year
jsonl_files_path = 'E:/twitter_data/extracted_geocoded_tweets/raw_jsonl/'

# initialise path variable citing output location for .csv files converted from .jsonl files
# this folder is to contain sub folders labelled by year
csv_files_path = 'E:/twitter_data/extracted_geocoded_tweets/raw_csv/'

# initialise list of years (formated as intergers) to extracted against
years_list = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]


# ### Load Administrative Geography Data
# ---
# The following three datasets were sourced from the ONS Open Geography portal (https://geoportal.statistics.gov.uk) and stored in a local directory specified above.
# 
# Source Data:
# 
#     - "C11OutputAreaLdn_region.shp": a .shp file containing vetor boundaries of all Output Areas in London
#     - "census-2011-oa-lookup-eng-0dd.xlsx": a .xlsx file containing a lookup table associating output areas with LA names 
#     - "LA Districts/LAD_MAY_2021_UK_BFC.shp": a .shp file containing vector boundaries of Local Authorities
# 

# In[ ]:


# load OA shape file as geodataframe
OA_gdf = gpd.read_file(admin_geo_path + "C11OutputAreaLdn_region.shp")

# load OA to LA lkup - lookup table categorising OA codes to LA names, also sourced from the ONS Open Geography portal
OALA_lkp = pd.read_excel(admin_geo_path + "census-2011-oa-lookup-eng-0dd.xlsx", sheet_name=LA )

# Load LA shape file as geodataframe - shape file defining LA vector boundaries
LA_boundary = gpd.read_file(admin_geo_path + "LA Districts/LAD_MAY_2021_UK_BFC.shp")


# ### Pre-processing Geospatial Data
# ---
# Once the relevant data have been loaded into the python environment it is must be pre-processed so that it is limited to the data that pertain to the LA of intrest and so that the Coordinate References System (CRS) used is standardised. 
# 
# The code below achieves this by carrying out the following process:
# 
#     - the LA boundary dataset is filtered to the LA of intrest
#     - the Output Area level data are enriched with LA Name values
#     - these values are used to filter the data to the LA for which a bounding box grid is being generated
#     - coordinate reference system data are standardises to EPSG: 27700

# In[ ]:


# filtering LA boundary dataset to LA of intrest
is_LA = LA_boundary['LAD21NM']==LA
LA_boundary = LA_boundary[is_LA]
LA_boundary = gpd.GeoDataFrame(LA_boundary, geometry='geometry')

# enrich OA data with LA name
LA_OA_gdf = pd.merge(
    OA_gdf, 
    OALA_lkp, 
    how="left", 
    left_on="OA11code", 
    right_on="OA11CD")

# filter OA data to LA of intrest
LA_OA_gdf = LA_OA_gdf.dropna()
LA_OA_gdf = gpd.GeoDataFrame(LA_OA_gdf, geometry='geometry')

# standardise CRS
OA_gdf = OA_gdf.set_crs(epsg=27700, allow_override=True)
LA_OA_gdf = LA_OA_gdf.set_crs(epsg=27700, allow_override=True)
LA_boundary = LA_boundary.set_crs(epsg=27700, allow_override=True)


# ### Generating bounding box grid covering LA Boundary
# ---
# 
# The code below generates a grid of specified resolution (box size) which has the same size and is positioned on the same coordinate reference system (CRS) as the LA boundary polygon processed above. It then filters that grid to those boxes that have one or more of their 4 corners falling within the LA boundary polygon.
# 
# This is done by implementing the following process:
# 
#     - variables are set for min and max x and y coordinates within the LA OA geodataframe using the .total_bounds attribute
#     - the desired height and width of boxes within the grid is specified
#     - a ceil value for number of rows and number of columns is calculated, the number of boxes required to fit the space
#     - a nested for loop is used to populate a list of box polygons, generated by iterating through the values set previously
#     - a 'grid' geodataframe is initialised and its CRS set to EPSG:27700
#     - a centroid value is added to each box (each row in the geodataframe) using the .centroid attribute
#     - the LA boundary polygon is merged into the geodataframe
#     - 'grid' geodataframe columns are renamed
#     - a function to return coordinates of a polygon as a list is defined and applied
#     - the geodataframe is exploded against these lists using the .explode() method
#     - the corner geopoints are flagged to indicate that they fall within the LA boundary polygon
#     - the geodataframe is filtered to those rows and de-duplciated to remove boxes represented multiple times

# In[ ]:


# set bounds as outer bounds of OA polygons
xmin,ymin,xmax,ymax =  LA_OA_gdf.total_bounds
# set height and width of boxes - this may be fine tuned to adjust the resolution of the grid and number of boxes
width = 250
height = 250
rows = int(np.ceil((ymax-ymin) /  height))
cols = int(np.ceil((xmax-xmin) / width))
XleftOrigin = xmin
XrightOrigin = xmin + width
YtopOrigin = ymax
YbottomOrigin = ymax- height
polygons = []
for i in range(cols):
    Ytop = YtopOrigin
    Ybottom =YbottomOrigin
    for j in range(rows):
        polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
        Ytop = Ytop - height
        Ybottom = Ybottom - height
    XleftOrigin = XleftOrigin + width
    XrightOrigin = XrightOrigin + width
grid = gpd.GeoDataFrame({'geometry':polygons})
grid = grid.set_crs(epsg=27700, allow_override=True)
#generate a column with a centorid value for each box
grid['box_centroid'] = grid['geometry'].centroid
#generate am LA column with the name of the LA
grid['LA'] = LA
#look up the geomtry of the LA boundary
grid = pd.merge(
    grid, 
    LA_boundary, 
    how="left", 
    left_on="LA", 
    right_on="LAD21NM")
grid.assign(LA_Boundary=LA)
grid = grid.filter(items=['LA','geometry_x', 'box_centroid', 'geometry_y', 'check'])
grid = grid.rename(columns={'geometry_x': 'box_polygon', 'geometry_y': 'LA_polygon'})
# initialise new dataframe to contain all boxes for over include logic
grid_inc = grid[['LA', 'box_polygon', 'LA_polygon']]
# define a function to return coordinates from a polygon as a list
def coord_lister(geom):
    coords = list(geom.exterior.coords)
    return (coords)
grid_inc['coord_list'] = grid_inc.box_polygon.apply(coord_lister)
# explode points within list of points, so df contains one row per each corner fo each box
grid_inc = grid_inc.explode('coord_list')
# split x and y coord values from tupels
grid_inc[['x_cord', 'y_cord']] = pd.DataFrame(grid_inc['coord_list'].tolist(), index=grid_inc.index)
# define each point in list as a geo point
grid_inc = gpd.GeoDataFrame(
    grid_inc, geometry=gpd.points_from_xy(grid_inc.x_cord, grid_inc.y_cord))
# set crs to EPSG:27700
grid_inc = grid_inc.set_crs(crs='EPSG:27700', allow_override=True)
# create 'check' column with Boolean indicating where points fall within the borough boundary
grid_inc['check'] = grid_inc['geometry'].within(grid_inc['LA_polygon'])
# filer to rows where 
grid_inc = grid_inc.loc[grid_inc['check'] == True]
# filter to required columns
grid_inc = grid_inc[['LA', 'box_polygon']]
# drop duplicates (single boxes across mutlipel rows, as they have more than one corner in the borough boundary)
grid_inc = grid_inc.drop_duplicates(subset=['box_polygon'])
# set 'box_polygon' as geometry
grid_inc = gpd.GeoDataFrame(grid_inc, geometry='box_polygon')


# ### Visualise bounding box gird for LA
# ---
# The bounding box gird generated above may be visualised on top of the OA and LA boundary polygons to ensure that it covers the LA boundary effectively. A count fo the total number of boxes within the grid is also given. This is important as gives an indication of the number of queries that will be required when extracting data from teh Twitter API. The previous step may be iterated through with different 'width' and 'height' values to alter the resolution of the grid.

# In[ ]:


# output count of bounding boxes within LA boundary
print("Count of bounding boxes: " + str(len(grid_inc)))
# visualise bounding boxes generated by over include logic (one or more corners)
fig, ax = plt.subplots(figsize=(25,25))
LA_OA_gdf.plot(ax = ax, color='none', edgecolor='grey')
LA_boundary.plot(ax = ax, color='none', edgecolor='black')
grid_inc.plot(ax = ax, color = 'red', edgecolor='none', alpha = 0.3, linewidth=1)
grid_inc.plot(ax = ax, color = 'none', edgecolor='white', alpha = 0.7, linewidth=1.5)


# ### Convert Bounding Box dataset from Eastings/Northings to Longitude/Latitude 
# ---
# Source data represent geospatial information using the British National Grid (BNG) coordinate referencing system. Geopoints are represented as Easting / Northing values (ESPG:27700). However, the Twitter API requires bounding box queries to represent geopoints as Longitude / Latitude values (ESPG:4326). Thus, it is necessary to convert the coordinates that we have generated. 
# 
# The code below achieves this by applying the following process:
# 
#     - a centroid value is derived for each bounding box
#     - a string formated version of the box polygon value is generated
#     - a series of all unique box polygon strings is created
#     - an array of sequential numbers of the same length is generated, assigning each box a number
#     - the series and array are combined into a dataframe
#     - a function is defined and applied to return coordinates from the polygon values as a list
#     - the dataframe is exploded against the geopoint lists
#     - geopoint tupels are then split to allow them to be fed into the transformer independantly
#     - the Transformer module of the pyproj library is imported 
#     - a transformer is intialised then applied within a loop to populate the 'latlons' list
#     - this 'latlons' list is then returned in on the grid_inc geodataframe, and lot/lon values split 
#     - the bbox_df dataframe consolidates the data into a one row per bounding box dataframe

# In[ ]:


# ignore pandas warnings
import warnings
warnings.simplefilter(action='ignore')

# derive box centroid
grid_inc['box_centorid'] = [x.centroid for x in grid_inc.box_polygon]

# create box polygon value formated as a string
grid_inc['box_polygon_str'] = [str(x) for x in grid_inc.box_polygon]

# create a series of unique string formated box polygon values
unique_box_str = pd.Series(grid_inc['box_polygon_str'].unique())

# create a series of sequential numbers (1,2,3,4 ...) the same length as the series of 
seq_arr = pd.Series(range(1,len(unique_box_str)+1))
box_no_df = pd.concat([unique_box_str, seq_arr], axis=1)
box_no_df = box_no_df.rename(columns={0:'box_polygon_str' , 1:'box_number'})
grid_inc = grid_inc.merge(box_no_df, left_on='box_polygon_str', right_on='box_polygon_str', how='left')

# define a function to return coordinates from a polygon as a list
def coord_lister(geom):
    coords = list(geom.exterior.coords)
    return (coords)
grid_inc['coords_of_box'] = grid_inc.box_polygon.apply(coord_lister)

# explode points within list of points, so df contains one row per each corner of each box
grid_inc = grid_inc.explode('coords_of_box')

## split x and y coord values from tupels
grid_inc[['x_cord', 'y_cord']] = pd.DataFrame(grid_inc['coords_of_box'].tolist(), index=grid_inc.index)

# define each point in list as a geo point
grid_inc = gpd.GeoDataFrame(
    grid_inc, geometry=gpd.points_from_xy(grid_inc.x_cord, grid_inc.y_cord))
grid_inc_coords = grid_inc[['x_cord', 'y_cord']]
grid_inc_coords['coords_of_box'] = grid_inc_coords.values.tolist()
coords_of_box_lists = grid_inc_coords.coords_of_box

# import Transformer module from the pyproj library
from pyproj import Transformer
transformer = Transformer.from_crs(27700, 4326)
latlons = []
for c in coords_of_box_lists:
    converted = transformer.transform(c[0], c[1])
    lat = converted[0]
    lon = converted[1]
    latlon = [lat, lon]
    latlons.append(latlon)

# arrange lat lons to new columns in dataframe
grid_inc['coords_of_box_latlon'] = latlons
grid_inc['lat'] = [x[0] for x in grid_inc.coords_of_box_latlon]
grid_inc['long'] = [x[1] for x in grid_inc.coords_of_box_latlon]
bbox_df = grid_inc[['box_number', 'lat', 'long']]
bbox_df = bbox_df.groupby(by='box_number').agg({'lat': ['min', 'max'], 'long': ['min', 'max']}, as_index=False)


# ### Reformatting bounding box data to enable use in Twitter API query
# ---
# 
# The Twitter API requires bounding box data to be formated in a particular way when executing bounding box based queries.
# 
# Require format: "bounding_box:[west_long south_lat east_long north_lat]"
#  
# Official guidance: https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
# 
# The code below impliments the following process to convert all bounding box gird data generated above into a string values that may be fed into quieries of the Twitter API:
# 
#     - list comprehension is used to return lists of the min and max lat and lon values for all boxes
#     - a for loop is used to iterate through a list derived from the index of the bbox_df dataframe
#     - the for loop arranges these the appropriate max and min lon and lat values inside the appropriate text
#     - each value is appended to an 'inputs' list
#     - the list is stored as a csv, then loaded back into the environment and visualised

# In[ ]:


# using list comprehension to create lists of min/max long/lats, ordered with box number
box_nos = [str(x) for x in bbox_df.index]
long_mins = [str(x) for x in bbox_df['long']['min']]
lat_mins = [str(x) for x in bbox_df['lat']['min']]
long_maxs = [str(x) for x in bbox_df['long']['max']]
lat_maxs = [str(x) for x in bbox_df['lat']['max']]

# iterating through the bbox numbers and appending min/max long/lats as strings
# along with outher characters required by the Twitter API
inputs = []
for x in box_nos:
    e = int(x) - 1
    min_long = long_mins[e]
    min_lat = lat_mins[e]
    max_long = long_maxs[e]
    max_lat = lat_maxs[e]
    input = str(x) + "|" + "bounding_box:[" + min_long + " " + min_lat + " "  + max_long + " " + max_lat + "]" 
    inputs.append(input)

# storing dataset of geo inputs as a csv
inputs_df = pd.DataFrame(inputs)
inputs_df.to_csv(bb_ouput_path + 'geo_input_list.csv', index=False)

# loading in csv to check
geo_inputs = pd.read_csv(bb_ouput_path + 'geo_input_list.csv').squeeze()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(geo_inputs)


# ## Extraction of Geocoded Tweets
# ---
# The code within this notebook takes an input a dataset of bounding box values and uses the Twarc2 Python package to extract Tweets made within those bounding boxes within specified years. The code then goes on to covert the extracted .jsonl files to .csv files in an equivalent folder substructure within a different specified directory. Having the data converted to csv allows for more ready analysis of the data to identify cohorts of intrest.
# 
# nb: Due to rate limits associated with the Twitter API (https://developer.twitter.com/en/docs/twitter-api/rate-limits) this code may take some time to execute depending on the number of years specified and the number of bounding boxes input.

# ### Load Bounding Boxes Data
# ---
# A dataset containing the bounding boxes values are loaded into the Python environment as a pandas dataframe. The format of these bounding boxes should be that required by the Twitter API with the addition of a numeric identifier and a pipe character as a pre-fix. The code below initialises the 'geo_lst' object as a list that will be iterated through to extract tweet data.

# In[ ]:


# load bounding box inputs and a pandas series
geo_inputs = pd.read_csv(bbox_data_path + 'geo_input_list.csv').squeeze()

# use list comprehension to generate a list of lists
# splitting on "|" to delineate between query input string and associated numeric identifier 
geo_lst = [x.split("|") for x in geo_inputs]


# ### Initialise Twarc2 client object using bearer token
# ---
# The code below sources a bearer token from a .txt file. This is a 114 character long alphanumeric string that uniquely identifies you as a user of the API and provisions access. Bearer tokens can be generated from within the Twitter Developer Portal (https://developer.twitter.com/en).

# In[ ]:


# initialise bearer_token variable
# from .txt file containing bearer token
with open(bearer_token_path + 'bearer_token.txt') as f:
    bearer_token = f.readlines()
bearer_token = bearer_token[0]

# initialise Twarc2 client object with bearer token
client = Twarc2(bearer_token=bearer_token)


# ### Twarc2 Query Extracting JSONL Files
# ---
# The code below extracts tweet data for tweets made within specified bounding boxes, within each specified year.
# 
# The code is executed within a for loop that iterates through each year interger within the 'years_list' list object. For each year 'y' a start time and end time are set in UTC format specifying the beginning of the first day of the year and the end of the last day of the year.
# 
# For each year, each bounding box in the 'geo_list' list object is iterated through. An output is then printed out specifying the year, bounding box number and bounding box input coordinates being 
# 
# Next a function is defined which executes an API call and stores the resultant data as a .jsonl file within the specified directory location. First the variable 'query' is initialised as the combination of the bounding box string plus the string '-is:retweet', which limits tweets sourced to those representing content generated by the user. Next, 'search_results' is initialised using the .search_all() method of the client object; with the query string, start tiem and end time fed in as inputs. Max_results is set to 100. This is the numebr of tweets per page, rathe rathan the total numebr of tweets that may be returned for each year / bounding box combination. The search_results generator class object which uses pagination to store tweet data. Each page within the object is then iterated through. For each page, a 'result' object is initialised using the .flatten() method of the Twarc2 expansion module and the page fed in as an input. This 'result' object is structured as a list of dictionaries containing all tweet data along with data pertaining to other entities such as the user and media attached to the tweet. Still within the function, a pseudo random interger is generated to serve as a unique identifier for the file and to facilitate any cross referencing between the python environment and the files stored within the folder structure. A .jsonl file is then opened in the appropriate year subfolder and the .dump() method of Python's json module is used to populate the .jsonl file with data from the 'result' object. The various inputs used are contained within the file name and encoding is set to utf8.
# 
# This function is then executed using the 'if __name__ == "__main__"' conditional block convention. 
# 
# nb: The code below will return tweet data for full years and must be amended to return tweet data for months within partial years.
# 
# nb: Due to rate limits associated with the Twitter API (https://developer.twitter.com/en/docs/twitter-api/rate-limits) this code may take some time to exicute depending on the number of years specified and the number of bounding boxes input.

# In[ ]:


# executing a for loop to iterate through the years specified above
for y in years_list:
    # specifying the start time in UTC for the period 
    start_time = datetime.datetime(y, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
    # specifying the end time in UTC for the period
    end_time = datetime.datetime(y, 12, 31, 23, 59, 59, 999999, datetime.timezone.utc)
    # execute a for loop iterating through the geo_lst list of bounding box input values
    for geo in geo_lst:
        # initialise variables for the bounding box numeric identifier and bounding box input string
        bb_no = geo[0]
        bb_input = geo[1]
        print("\nYear: " + str(y))
        print("Bounding Box Number: " + bb_no)
        print("Bounding Box Input: " + bb_input)
        # define main() function executing a query of the Twitter API
        def main():
            # initialise query variable as bb_input
            query = bb_input + '-is:retweet'
            # using the Twarc2 search_all method to call the full-archive search endpoint 
            # to get Tweets based on the query, start and end times
            search_results = client.search_all(query=query, start_time=start_time, end_time=end_time, max_results=100)
            # tweets returned using pagination, so iterate through pages of search results
            for page in search_results:
                # expansions.flatten used to flattern data pertaining to the tweet and associated entities (user, media etc.)
                result = expansions.flatten(page)
                # initialise pseudo random id for file to allow for files to be uniquely identifiable
                id = random.randint(1000000000000,9999999999999)
                print("File ID: " + str(id))
                # store extracted tweets as a .jsonl file in the specified directory
                with open(jsonl_files_path + str(y) + '/' + str(id) + '_' + str(y) + '_bb' + str(bb_no) + '.jsonl',
                          'w', encoding ='utf8') as f:
                    json.dump(result, f, ensure_ascii=True)
        # execute main() function using if __name__ == "__main__" conditional block                         
        if __name__ == "__main__":
            main()


# ### Initialisation of DataFrame Converter
# ---
# The code below intitialises a DataFrame conversion object using the DataFrameConverter class of Twarc2's DataFrameConverter module.  

# In[ ]:


# initialise dfconverter object using Twarc2's DataFrameconverter class
dfconverter = DataFrameConverter(
    input_data_type="tweets",
    json_encode_all=True,
    json_encode_text=True,
    json_encode_lists=True,
    inline_referenced_tweets=False,
    merge_retweets=False,
    allow_duplicates=True,
)


# ### Conversion of JSONL files to CSV files
# ---
# The code below carries out a process by which all .jsonl files extracted via the process above are converted into .csv files. The .csv files are to be stored into a year folder structure equivalent to that of the .jsonl files. The pseudo random numeric identifier is retained to associate each .csv file with its associated .jsonl file to facilitate debugging in the even that teh process falls over. The process below is designed to be iterative in that it will only convert files that are yet to be converted. This means that in the event that the process falls over, fixes can be made and the process resumed without having to reconvert files.

# In[ ]:


# initialise list of year folders within 'raw_json' folder
yr_folders = os.listdir(jsonl_files_path)
# for loop iterating through each of the year folders in the 'raw_json' folder
for yr_f in yr_folders:
    # initialising variables as list objects
    # containing string values of file names of all files in specified folders
    jsonl_list = os.listdir(jsonl_files_path + yr_f + '/') 
    csv_list = os.listdir(csv_files_path + yr_f + '/')
    # list comprihension used to initialise list object replacing '.csv' suffex with '.jsonl' suffex
    # to allow for comparison of lists
    csv_list = [x.replace('.csv', '.jsonl') for x in csv_list]
    # list comprehension used to intialise list object 
    # containing string values of all file names of .jsonl files yet to be converted
    file_list = [x for x in jsonl_list if x not in csv_list]
    # printing out a summary of the files to be converted in the rae_json and raw_csv folders for the year
    print('\nYear Folder:' + str(yr_f))
    print('Total .jsonl files: ' + str(len(jsonl_list)))
    print('Files successfully converted so far: ' + str(len(csv_list)))
    print('Files left to convert: ' + str(len(file_list)))
    # for loop iterating through the file_list list of all .jsonl files yet to be converted
    for f in file_list:
        # opening a specifield .jsonl file
        with open(jsonl_files_path + str(y) + "/" + f , "r") as infile:
            # using .replace() method to remove '.jsonl' suffex from file name string
            f = f.replace('.jsonl','')  
            # opening a .csv files 
            with open(csv_files_path + str(y) + "/" + f + ".csv", "w") as outfile: 
                # initialising converter using Twarc2's CSVConverter class
                # with inputs specified
                converter = CSVConverter(infile=infile, outfile=outfile,  converter=dfconverter)    
                # processing conversion using the .process() method
                converter.process()    
                # output confirmation of completed conversion
                print(f + ' - generated')

