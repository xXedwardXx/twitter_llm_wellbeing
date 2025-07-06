#!/usr/bin/env python
# coding: utf-8

# # ANALYSIS OF LLM INFERENCE DATA
# ---
# This notebook contains a data analytics pipeline that consumes JSON data held against the following keys: 'concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'languages', 'hap_lvl', 'hap_conf', 'hap_topics', 'hap_explain', 'anx_lvl', 'anx_conf', 'anx_topics',  'anx_explain', 'emotions', 'glc_topics'.
# 
# These data were output by a Large Language Model (GPT4o-mini), which was prompted to make inferences on user/month blocks of tweet data.
# 
# The pipeline:
# - loads data from JSON source files
# - carries out simple EDA and calculates some summary statistics
# - performs pre-processing, filtering based on confidence levels, and cleansing topic data
# - cross-references LLM-derived mean happiness and anxiety levels against published ONS statistics
# - performs month-on-month longitudinal analysis of happiness and anxiety levels
# - word cloud visualisations
# - TF-IDF experimentation (not included in final analysis)
# - statistical modelling of data aggregated to a monthly prevalence measure
# - visualisation

# # LOAD LIBRARIES AND DATA
# ---

# ## Import Libraries

# In[1]:


import os
import json
import math
import string
import gc
import re
import calendar
import datetime
import random
import pyarrow
import openpyxl
from collections import Counter
from collections import OrderedDict
from itertools import product
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.pyplot import plot, savefig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch
from highlight_text import fig_text, ax_text
from wordcloud import (WordCloud, get_single_color_func)
from wordcloud import WordCloud
from PIL import Image
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import nltk.data
from Levenshtein import distance as levenshtein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import scipy.stats as stats
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from scipy.stats import f

import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.formula.api import ols
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

from fuzzywuzzy import fuzz

import ruptures as rpt


# ## Initial Load, consolidation and storage of data (run once)
# 

# In[ ]:


## DEFINE FUNCTION TO LOAD DATA FROM MANY JSON FILES
def load_json_files_to_dataframe(folder_path):
    """
    Loads all JSON files in the specified folder into a single Pandas DataFrame.
    The DataFrame will include a column for the filename and one column for each key in the JSON objects.
    Parameters:
        folder_path (str): Path to the folder containing JSON files.
    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all JSON files.
    """
    all_data = []
    for filename in os.listdir(folder_path):  # Iterate over all files in the folder
        if filename.endswith('.json'):  # Only process JSON files
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)  # Load JSON data
                    data = json.loads(data)
                    if isinstance(data, dict):  # JSON is a single dictionary                    
                        data["filename"] = filename
                        data["concat_key"]  = filename.replace('_', '|').replace('.json', '')
                        data_as_str = str(data)
                        #token_count = o200k_tokens_from_string(data_as_str)
                        #data["output_token_count"] = token_count                 
                        all_data.append(data)    
                    elif isinstance(data, list):  # JSON is a list of records
                        for record in data:
                            if isinstance(record, dict):  # Ensure each record is a dictionary
                                record["filename"] = filename
                                all_data.append(record)
                            else:
                                print(f"Skipping non-dict record in {filename}: {record}")
                    elif isinstance(data, str):  # JSON is a single string
                        all_data.append({"filename": filename, "content": data})
                    else:  # JSON is neither a list, dictionary, nor string
                        print(f"Skipping unsupported JSON structure in {filename}: {type(data)}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        data["concat_key"] = filename.replace('_', '|').replace('.json', '')
    # Convert the list of data into a Pandas DataFrame
    return pd.DataFrame(all_data)


# In[ ]:


## LOAD DATA FROM JSON FILES INTO SINGLE DATAFRAME
folder_path = 'E:/twitter_analysis_data/llm_outputs/jan_03_full_extraction/'
df = load_json_files_to_dataframe(folder_path)


# In[ ]:


## DERIVING VARIABLES FROM CONCAT KEY 
# snsure usr_mnth_df is a copy, not a slice
df = df.copy()
# split 'concat_key' and assign directly
split_cols = df['concat_key'].str.split('|', expand=True)
split_cols.columns = ['year', 'month_num', 'user_id']
df.loc[:, 'year'] = pd.to_numeric(split_cols['year'], errors='coerce')
df.loc[:, 'month_num'] = pd.to_numeric(split_cols['month_num'], errors='coerce').fillna(0).astype(int)
df.loc[:, 'user_id'] = split_cols['user_id'].astype(str)
# Map month abbreviations and names
month_abbrs = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df.loc[:, 'month_abbr'] = df['month_num'].map(month_abbrs)
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
              7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
df.loc[:, 'month_name'] = df['month_num'].map(month_names)
# create mnth_year_txt variable
df['mnth_yr_text'] = df['month_name'] + " " + df['year'].astype(str)
# tweaking column order and dropping file_name
col_order = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text',
             'languages',
             'hap_lvl', 'hap_conf', 'hap_topics','hap_explain', 
             'anx_lvl', 'anx_conf', 'anx_topics', 'anx_explain',
             'emotions', 'glc_topics']
df = df[col_order]


# In[ ]:


# STORE DATA TO SINGLE JSON FILE
path = "E:/twitter_analysis_data/llm_outputs/full_extraction_consolidated_data/"
df.to_json(path + "full_data_json.json", orient="records", lines=True)


# ## Load data from single file

# In[2]:


# LOAD DATA FROM SINGLE JSON FILE
path = "E:/twitter_analysis_data/llm_outputs/full_extraction_consolidated_data/"
df_json = pd.read_json(path + "full_data_json.json", orient="records", lines=True)
df_json['user_id'] = df_json['user_id'].astype(str)
df = df_json
del df_json


# # EXPLORATORY DATA ANALYSIS
# ---

# ## Basic EDA

# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.sample(3)


# ## Summary Statistics

# In[ ]:


## summary stats:
print("SUMMARY STATISTICS:")
# total user/month blocks
print("Total user / month blocks: " + str(len(df['concat_key'])))
# unique users
print("Total unique users: " + str(df['user_id'].nunique(dropna=False)))
# unique months
print("Total unique months: " + str(df['mnth_yr_text'].nunique(dropna=False)))

# Count occurrences of concat_key for each month-year combination
monthly_counts = df.groupby(['year', 'month_num']).size().reset_index(name='count')
# Convert month numbers to readable format
monthly_counts['month'] = monthly_counts['month_num'].apply(lambda x: f'{x:02d}')
# Create a year-month column for plotting
monthly_counts['year_month'] = monthly_counts['year'].astype(str) + '-' + monthly_counts['month']
# Convert year_month to datetime format for proper chronological sorting
monthly_counts['year_month'] = pd.to_datetime(monthly_counts['year_month'], format='%Y-%m')
# Sort the data by date
monthly_counts = monthly_counts.sort_values(by='year_month')

print("Maximum unique users within month: " + str(monthly_counts['count'].max()))
print("Minimum unique users within month: " + str(monthly_counts['count'].min()))

## Defining counting function to count all items, and unique items, in all lists within a df column as a series
def count_items_in_lists_col(series):
    list_of_lists = []
    for x in series:    
        for item in x:
            list_of_lists.append(item)        
    total_items = len(list_of_lists)
    unique_items = len(set(list_of_lists))
    print("Total items: " +  str(total_items))
    print("Unique items: " + str(unique_items))

# Languages
print("\n\033[1mLanguages:\033[0m")
count_items_in_lists_col(df['languages'])

# Happiness Topics
print("\n\033[1mHappiness Topics:\033[0m")
count_items_in_lists_col(df['hap_topics'])
# Anxiety Topics
print("\n\033[1mAnxiety Topics:\033[0m")
count_items_in_lists_col(df['anx_topics'])
# Emotions
print("\n\033[1mEmotions:\033[0m")
count_items_in_lists_col(df['emotions'])
# Emotions
print("\n\033[1mGood Life Camden Topics:\033[0m")
count_items_in_lists_col(df['glc_topics'])


# ## Summary of Numeric Data

# In[ ]:


# SUMMARY OF NUMERIC DATA
# Summary statistics for numerical columns
numerical_summary = df.describe()
# Distribution of happiness and anxiety levels
hap_anx_distribution = df[['hap_lvl', 'anx_lvl']].value_counts().reset_index(name='count')
# Yearly trend of happiness and anxiety levels
yearly_trend = df.groupby('year')[['hap_lvl', 'anx_lvl']].mean().reset_index()
numerical_summary


# In[ ]:


## VISULATISATION OF HAP ANX DISTRIBUTION
# Create a pivot table for the cross table heatmap
hap_anx_pivot = df.pivot_table(index='hap_lvl', columns='anx_lvl', aggfunc='size', fill_value=0)
# Create a custom colormap where 0 values are white
cmap = plt.cm.YlOrBr  # Original colormap
cmap_colors = cmap(np.arange(cmap.N))  # Extract colors
# Set the first color (corresponding to 0 values) to white
cmap_colors[0] = [1, 1, 1, 1]  # RGBA for white
custom_cmap = mcolors.ListedColormap(cmap_colors)
# Plot heatmap using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
# Create heatmap with the custom colormap and white for 0 values
cax = ax.imshow(hap_anx_pivot, cmap=custom_cmap, aspect='auto')
# Add colorbar
fig.colorbar(cax, ax=ax)
# Set labels and title
ax.set_xlabel("Anxiety Level")
ax.set_ylabel("Happiness Level")
ax.set_title("Happiness vs. Anxiety Levels within a User/Month", loc="left", fontsize=14)
# Set ticks and labels
ax.set_xticks(np.arange(len(hap_anx_pivot.columns)))
ax.set_yticks(np.arange(len(hap_anx_pivot.index)))
ax.set_xticklabels(hap_anx_pivot.columns)
ax.set_yticklabels(hap_anx_pivot.index)
# Show values in each cell
for i in range(len(hap_anx_pivot.index)):
    for j in range(len(hap_anx_pivot.columns)):
        color = 'black' if hap_anx_pivot.iloc[i, j] != 0 else 'white'  # Make text invisible on white cells
        ax.text(j, i, hap_anx_pivot.iloc[i, j], ha='center', va='center', color=color)
# Show plot
plt.show()


# In[ ]:


## Test Hap lvl Anx Levl correlation
lvls_df = df[['hap_lvl','anx_lvl']].copy()
# Compute Pearson's Correlation and p-value using SciPy
lvls_correlation = lvls_df.corr(method='pearson')
corr_coef, p_value = pearsonr(lvls_df["hap_lvl"], lvls_df["anx_lvl"])
print(f"Pearson Correlation Coefficient: {corr_coef}")
print(f"P-value: {p_value}")
# Compute Kendall's Tau and p-value using SciPy
tau, p_value = kendalltau(lvls_df["hap_lvl"], lvls_df["anx_lvl"])
print(f"\nKendall’s Tau: {tau}")
print(f"P-value: {p_value}")


# In[ ]:


## VISULATISE DISTRIBUTION OF HAP/ANX LEVELS AND CONFIDENCE
# Create a histogram for hap_lvl and anx_lvl with exactly 10 bins
plt.figure(figsize=(6, 6))
# Plot happiness levels with mustard yellow color
plt.hist(df['hap_lvl'], bins=10, color='#FFC000', alpha=0.3, label='Happiness Level')  # Mustard Yellow
# Plot anxiety levels with grey color
plt.hist(df['anx_lvl'], bins=10, color='grey', alpha=0.3, label='Anxiety Level')
# Labels and title
#plt.xlabel("Level")
#plt.ylabel("Frequency")
plt.title("Distribution of Happiness and Anxiety Scores", loc='left', fontsize=13)
plt.legend()
plt.grid(True)
# Ensure all x-axis labels (0 to 10) are present
plt.xticks(range(11))
# Show plot
plt.show()
# Plot distributions for happiness and anxiety confidence
plt.figure(figsize=(6, 6))
df['hap_conf'].hist(bins=10, label='Happiness Level Confidence', color='#FFC000', alpha=0.3)
df['anx_conf'].hist(bins=10, label='Anxiety Level Confidence', color='grey', alpha=0.3)
plt.title('Distribution of Confidence Levels', loc='left', fontsize=13)
#plt.xlabel('Confidence Scores')
#plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[ ]:


## EXPLORE CORRELATIONS BETWEEN HAPPINESS/ANXIETY LEVELS AND ASSOCIATED CONFIDENCE LEVELS
# Extract the two relevant variables
x = df['hap_lvl']
y = df['hap_conf']
# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]
# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.2, color='#FFC000')
# Add title and labels
plt.xlabel("Happiness Level")
plt.ylabel("Happiness Confidence")
plt.title(f"Scatter Plot of Happiness Confidence vs. Happiness Levels\nCorrelation: {correlation:.2f}", loc='left')
# Show plot
plt.show()
# Extract the two relevant variables
x = df['anx_lvl']
y = df['anx_conf']
# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]
# Create scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(x, y, alpha=0.2, color='grey')
# Add title and labels
plt.xlabel("Anxiety Level")
plt.ylabel("Anxiety Confidence")
plt.title(f"Scatter Plot of Anxiety Confidence vs. Anxiety Levels\nCorrelation: {correlation:.2f}", loc='left')
# Show plot
plt.show()


# # PRE-PROCESSING
# ---

# ## Derive Financial Year Varaible

# In[ ]:


### DERIVE FINANCIAL YEAR VARIABLE
# derive 'financial year' field
df["financial_year"] = df.apply(
    lambda row: f"{row['year']}-{row['year']+1}" if row["month_num"] >= 4 else f"{row['year']-1}-{row['year']}",axis=1)
fin_yr_map = {"2017-2018":"2017-18", "2018-2019":"2018-19", "2019-2020":"2019-20", "2020-2021":"2020-21", "2021-2022":"2021-22", "2022-2023":"2022-23"}
df["financial_year"] = df["financial_year"].map(fin_yr_map)


# ## Confidence level filtering    

# In[ ]:


# APPLY CONFIDENCE LEVEL FILTERING
original_rows = df.shape[0]
df = df[(df["hap_conf"] >= 0.7) & (df["anx_lvl"] >= 0.7)]
rows_after_filering = df.shape[0]
rows_remove = original_rows - rows_after_filering
print("Rows removed due to confidence level filtering: " + str(rows_remove))


# ## Cleansing of Topic Data   

# In[ ]:


## MODEL AND CLEANSE LANGUAGE DATA
# create languages exploded df
lang_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'languages']
lang_df = df[lang_cols].copy() #create a new dataframe with lang lists and id/time columns
lang_df = lang_df.explode("languages") #explode language lists
lang_df['languages_cleansed'] = [s.title() for s in lang_df['languages']] ### apply standard formatting (capitalisation)
lang_df['languages_cleansed'] = [s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for s in lang_df['languages_cleansed']] # replace punctuation with space
lang_df['languages_cleansed'] = [' '.join(s.split()) for s in lang_df['languages_cleansed']] # trim spaces
lang_df = lang_df[~lang_df['languages_cleansed'].str.lower().isin(['nan', 'none'])]
# create groupby count
lang_grouped = lang_df.groupby('languages_cleansed').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# create list of unique values
unique_lang_values = list(lang_df['languages_cleansed'].unique())

## SUMMARISE LANGUAGES
# print summary info
print("\033[1mLanguages Summary\033[0m")
print("Total identifications: " + str(lang_df.shape[0]))
print("Unique Languages identified: " + str(len(unique_lang_values)))
print("Unique users: " +  str(lang_df['user_id'].nunique()))
print("Unique months: " +  str(lang_df['mnth_yr_text'].nunique()))
print("Unique users/months: " +  str(lang_df['concat_key'].nunique()))
# plot counts of top 100
lang_grouped_top_100 = lang_grouped.sort_values(by='Count', ascending=False).head(100).sort_values(by='Count', ascending=True)
plt.figure(figsize=(11, 21))
plt.title('Top 100 Languages')
plt.barh(lang_grouped_top_100['languages_cleansed'], lang_grouped_top_100['Count'], height=0.5, color='teal', alpha=0.5)
plt.yticks(fontsize=8)
plt.margins(y=0.01)
plt.show()


# In[ ]:


## MODEL AND CLEANSE HAPPINESS TOPICS
# create happiness topics exploded df
hap_top_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'hap_topics']
hap_top_df = df[hap_top_cols].copy() #create a new dataframe with happiness topic lists and id/time columns
hap_top_df = hap_top_df.explode("hap_topics") #explode language lists
hap_top_df['hap_topics'] = [str(s) for s in hap_top_df['hap_topics']] #ensure string format
hap_top_df = hap_top_df.rename(columns={'hap_topics':'hap_topics_raw'}) #rename to retain raw format

## SIMPLE PRE-PROCESSING
# set to lowercase
hap_top_df['hap_topics_cleansed'] = [s.lower() for s in hap_top_df['hap_topics_raw']] #convert all characters to lower case
hap_top_df = hap_top_df[~hap_top_df['hap_topics_cleansed'].isin(['nan', 'none'])] #remove rows containing 'nan' or 'none'
# remove punctuation
hap_top_df['hap_topics_cleansed'] = [s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for s in hap_top_df['hap_topics_cleansed']]
# remove " s " left from punctuation removal
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\s+s\s+', ' ', regex=True)
# Remove all numeric characters
hap_top_df['hap_topics_cleansed']  = hap_top_df['hap_topics_cleansed'] .str.replace(r'\d+', '', regex=True)

## STOP WORD REMOVAL
def remove_a_the(text):
    return re.sub(r'\b(?:a|the)\b(?!\w*a\w*)', '', text, flags=re.IGNORECASE).strip()
# apply function to the column
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].apply(remove_a_the)

##LEMTIZATION
# initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = word_tokenize(text)  # Tokenize into words
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]  # Lemmatize each word
    return ' '.join(lemmatized_words)  # Join back into a string
# apply lemmatization to the column
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].apply(lemmatize_text)

## MANUAL TERM SUBSTITUTION
#standardise common words/terms#
#hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bcovid[\s]?19\b', 'covid', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bcovid[-_ ]?19\b', 'covid', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bhouse\b', 'housing', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bfor\b', 'of', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\blearn\b', 'learning', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bread\b', 'reading', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bcook\b', 'cooking', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bserice\b', 'services', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\binteractions\b', 'interaction', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bconnections\b', 'connection', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bwell be\b', 'wellbeing', regex=True)
hap_top_df['hap_topics_cleansed'] = hap_top_df['hap_topics_cleansed'].str.replace(r'\bservice\b', 'services', regex=True)

# trim spaces
hap_top_df['hap_topics_cleansed'] = [' '.join(s.split()) for s in hap_top_df['hap_topics_cleansed']]

## FURTHER MODELING TO GROUPED COUNTS
# create list of unique values
unique_hap_top_values = list(hap_top_df['hap_topics_cleansed'].unique())
# print for checking
print("hap_top (pre merge) shape: ", hap_top_df.shape)
# create raw groupby count
hap_top_raw_grouped = hap_top_df.groupby('hap_topics_raw').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# create cleansed groupby count
hap_top_cleansed_grouped = hap_top_df.groupby('hap_topics_cleansed').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# merge raw and cleansed groupby counts for review
hap_top_df = hap_top_df.merge(hap_top_raw_grouped, on='hap_topics_raw',how='left')
hap_top_df.rename(columns={'Count': 'raw_term_count'}, inplace=True)
hap_top_df = hap_top_df.merge(hap_top_cleansed_grouped, on='hap_topics_cleansed',how='left')
hap_top_df.rename(columns={'Count': 'cleansed_term_count'}, inplace=True)
hap_top_df.sort_values(by=['cleansed_term_count','hap_topics_cleansed', 'raw_term_count', 'hap_topics_raw']
                       , ascending=[False, False, False, False], inplace=True)
# print for checking
print("RAW / CLEANSED / MERGED DF SHAPE CHECK:")
print("hap_top_raw_grouped shape: ", hap_top_raw_grouped.shape)
print("hap_top_cleansed_grouped shape: ", hap_top_cleansed_grouped.shape)
print("hap_top (post merge) shape: ", hap_top_df.shape)
hap_top_check_df = hap_top_df[['hap_topics_raw', 'hap_topics_cleansed','raw_term_count', 'cleansed_term_count']]
hap_top_check_df = hap_top_check_df.drop_duplicates(subset=['hap_topics_raw'])
# print for checking
print("hap_top_check_df shape: ", hap_top_check_df.shape)

## SUMMARISE HAPPINESS TOPICS
# print summary info
print("\033[1mHappiness Topics Summary\033[0m")
print("Total identifications: " + str(hap_top_df.shape[0]))
print("Unique Topics: " + str(len(unique_hap_top_values)))
print("Unique users: " +  str(hap_top_df['user_id'].nunique()))
print("Unique months: " +  str(hap_top_df['mnth_yr_text'].nunique()))
print("Unique users/months: " +  str(hap_top_df['concat_key'].nunique()))
# plot counts of top 500 cleansed
hap_top_grouped_top_500 = hap_top_cleansed_grouped.sort_values(by='Count', ascending=False).head(500).sort_values(by='Count', ascending=True)
plt.figure(figsize=(15, 60))
plt.title('Top 500 Happiness Topics')
plt.barh(hap_top_grouped_top_500['hap_topics_cleansed'], hap_top_grouped_top_500['Count'], height=0.8, color='teal', alpha=0.5)
plt.yticks(fontsize=8)
plt.margins(y=0.005)
plt.show()


# In[ ]:


## MODEL ANXIETY TOPICS
# create happiness topics exploded df
anx_top_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'anx_topics']
anx_top_df = df[anx_top_cols].copy() #create a new dataframe with happiness topic lists and id/time columns
anx_top_df = anx_top_df.explode("anx_topics") #explode language lists (reset index??)
anx_top_df['anx_topics'] = [str(s) for s in anx_top_df['anx_topics']] #ensure string format
anx_top_df = anx_top_df.rename(columns={'anx_topics':'anx_topics_raw'}) #rename to retain raw format

###SIMPLE PRE-PROCESSING
# set to lowercase
anx_top_df['anx_topics_cleansed'] = [s.lower() for s in anx_top_df['anx_topics_raw']] #convert all characters to lower case
anx_top_df = anx_top_df[~anx_top_df['anx_topics_cleansed'].isin(['nan', 'none'])] #remove rows containing 'nan' or 'none'
#remove punctuation
anx_top_df['anx_topics_cleansed'] = [s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for s in anx_top_df['anx_topics_cleansed']]
# remove " s " left from punctuation removal
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\s+s\s+', ' ', regex=True)
# Remove all numeric characters
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\d+', '', regex=True)

## STOPWORD REMOVAL
def remove_a_the(text):
    return re.sub(r'\b(?:a|the)\b(?!\w*a\w*)', '', text, flags=re.IGNORECASE).strip()
# apply function to the column
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].apply(remove_a_the)

## LEMTIZATION
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = word_tokenize(text)  # Tokenize into words
    lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]  # Lemmatize each word
    return ' '.join(lemmatized_words)  # Join back into a string
# Apply lemmatization to the column
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].apply(lemmatize_text)

## MANUAL TERM SUBSTITUTION
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\btime\b', 'times', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bcovid[-_ ]?19\b', 'covid', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'-related\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'-relate\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r' relate\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r' stress\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r' issue\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r' concern\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r' crisis\b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'concern about \b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'anxiety about \b', '', regex=True).str.strip()
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bcost of live\b', 'cost of living', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bhouse\b', 'housing', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bfor\b', 'of', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bservice\b', 'services', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\binteractions\b', 'interaction', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bconnections\b', 'connection', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace(r'\bcelebrations\b', 'celebration', regex=True)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('political', 'politics', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('work relate', 'work', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('well be', 'wellbeing', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('crime rat', 'crime rate', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('live condition', 'living conditions', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('vaccination rat', 'vaccination rate', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('covid infection rat', 'covid infection rate', regex=False)
anx_top_df['anx_topics_cleansed'] = anx_top_df['anx_topics_cleansed'].str.replace('fundraise', 'fundraising', regex=False)

# trim spaces
anx_top_df['anx_topics_cleansed'] = [' '.join(s.split()) for s in anx_top_df['anx_topics_cleansed']]

## FURTHER MODELING TO GROUP COUNTS
# create a list of unique values
unique_anx_top_values = list(anx_top_df['anx_topics_cleansed'].unique())
# print for checking
print("anx_top (pre merge) shape: ", anx_top_df.shape)
# create raw groupby count
anx_top_raw_grouped = anx_top_df.groupby('anx_topics_raw').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# create cleansed groupby count
anx_top_cleansed_grouped = anx_top_df.groupby('anx_topics_cleansed').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# merge raw and cleansed groupby counts for review
anx_top_df = anx_top_df.merge(anx_top_raw_grouped, on='anx_topics_raw',how='left')
anx_top_df.rename(columns={'Count': 'raw_term_count'}, inplace=True)
anx_top_df = anx_top_df.merge(anx_top_cleansed_grouped, on='anx_topics_cleansed',how='left')
anx_top_df.rename(columns={'Count': 'cleansed_term_count'}, inplace=True)
anx_top_df.sort_values(by=['cleansed_term_count','anx_topics_cleansed', 'raw_term_count', 'anx_topics_raw']
                       , ascending=[False, False, False, False], inplace=True)
# print for checking
print("RAW / CLEANSED / MERGED DF SHAPE CHECK:")
print("anx_top_raw_grouped shape: ", anx_top_raw_grouped.shape)
print("anx_top_cleansed_grouped shape: ", anx_top_cleansed_grouped.shape)
print("anx_top (post merge) shape: ", anx_top_df.shape)
anx_top_check_df = anx_top_df[['anx_topics_raw', 'anx_topics_cleansed','raw_term_count', 'cleansed_term_count']]
anx_top_check_df = anx_top_check_df.drop_duplicates(subset=['anx_topics_raw'])
# print for checking
print("anx_top_check_df shape: ", anx_top_check_df.shape)

## SUMMARISE HAPPINESS TOPICS
# print summary info
print("\033[1mAnxiety Topics Summary\033[0m")
print("Total identifications: " + str(anx_top_df.shape[0]))
print("Unique Topics: " + str(len(unique_anx_top_values)))
print("Unique users: " +  str(anx_top_df['user_id'].nunique()))
print("Unique months: " +  str(anx_top_df['mnth_yr_text'].nunique()))
print("Unique users/months: " +  str(anx_top_df['concat_key'].nunique()))
# plot counts of top 500 cleansed
anx_top_grouped_top_500 = anx_top_cleansed_grouped.sort_values(by='Count', ascending=False).head(500).sort_values(by='Count', ascending=True)
plt.figure(figsize=(15, 60))
plt.title('Top 500 Anxiety Topics')
plt.barh(anx_top_grouped_top_500['anx_topics_cleansed'], anx_top_grouped_top_500['Count'], height=0.8, color='teal', alpha=0.5)
plt.yticks(fontsize=8)
plt.margins(y=0.005)
plt.show()


# In[ ]:


## MODEL AND CLEANSE EMOTIONS
# create happiness topics exploded df
emo_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'emotions']
emo_df = df[emo_cols].copy() #create a new dataframe with happiness topic lists and id/time columns
emo_df = emo_df.explode("emotions") #explode language lists
emo_df = emo_df.rename(columns={'emotions':'emotions_raw'})

### apply standard formatting to consolidate against minor differences
emo_df['emotions_cleansed'] = [str(s) for s in emo_df['emotions_raw']] #ensure string format
emo_df['emotions_cleansed'] = [s.lower() for s in emo_df['emotions_cleansed']] #convert all characters to lower case
emo_df['emotions_cleansed'] = [s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for s in emo_df['emotions_cleansed']] # replace punctuation with space
emo_df['emotions_cleansed'] = [' '.join(s.split()) for s in emo_df['emotions_cleansed']] # trim spaces
emo_df = emo_df[~emo_df['emotions_cleansed'].str.lower().isin(['nan', 'none'])]

# create groupby count
emo_grouped = emo_df.groupby('emotions_cleansed').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# create list of unique values
unique_emo_values = list(emo_df['emotions_cleansed'].unique())

## SUMMARISE HAPPINESS TOPICS
# print summary info
print("\033[1mEmotions Summary\033[0m")
print("Total identifications: " + str(emo_df.shape[0]))
print("Unique Emotions: " + str(len(unique_emo_values)))
print("Unique users: " +  str(emo_df['user_id'].nunique()))
print("Unique months: " +  str(emo_df['mnth_yr_text'].nunique()))
print("Unique users/months: " +  str(emo_df['concat_key'].nunique()))
# plot counts of top 100
emo_grouped_top_100 = emo_grouped.sort_values(by='Count', ascending=False).head(100).sort_values(by='Count', ascending=True)
plt.figure(figsize=(12, 21))
plt.title('Top 100 Emotions')
plt.barh(emo_grouped_top_100['emotions_cleansed'], emo_grouped_top_100['Count'], height=0.5, color='teal', alpha=0.5)
plt.yticks(fontsize=8)
plt.margins(y=0.01)
plt.show()


# In[ ]:


## MODEL AND CLEANSE GLC TOPICS
# create happiness topics exploded df
glc_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'glc_topics']
glc_df = df[glc_cols].copy() #create a new dataframe with happiness topic lists and id/time columns
glc_df = glc_df.explode("glc_topics") #explode language lists
glc_df = glc_df.rename(columns={'glc_topics':'glc_topics_raw'})

### apply standard formatting to consolidate against minor differences
glc_df['glc_topics_cleansed'] = [str(s) for s in glc_df['glc_topics_raw']] #ensure string format
glc_df['glc_topics_cleansed'] = [s.lower() for s in glc_df['glc_topics_cleansed']] #convert all characters to lower case
glc_df['glc_topics_cleansed'] = [s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for s in glc_df['glc_topics_cleansed']] # replace punctuation with space
glc_df['glc_topics_cleansed'] = [' '.join(s.split()) for s in glc_df['glc_topics_cleansed']] # trim spaces
glc_df = glc_df[~glc_df['glc_topics_cleansed'].isin(['nan', 'none'])]

# filter to specified topics
glc_tops = ['cultural identity', 'local spaces', 'local services', 'community connections', 'housing', 'employment', \
 'personal finances', 'local environment', 'feelings of safety', 'learning opportunities', 'educational outcomes']
glc_df = glc_df[glc_df['glc_topics_cleansed'].isin(glc_tops)]

# create groupby count
glc_grouped = glc_df.groupby('glc_topics_cleansed').size().reset_index(name='Count').sort_values(by='Count', ascending=False)
# create list of unique values
unique_glc_values = list(glc_df['glc_topics_cleansed'].unique())

## SUMMARISE HAPPINESS TOPICS
# print summary info
print("\033[1mGood Life Camden Topics Summary\033[0m")
print("Total rows in exploded df(total identifications): " + str(glc_df.shape[0]))
print("Unique Topics: " + str(len(unique_glc_values)))
print("Unique users: " +  str(glc_df['user_id'].nunique()))
print("Unique months: " +  str(glc_df['mnth_yr_text'].nunique()))
print("Unique users/months: " +  str(glc_df['concat_key'].nunique()))
# plot counts
glc_grouped = glc_grouped.sort_values(by='Count', ascending=False).head(100).sort_values(by='Count', ascending=True)
plt.figure(figsize=(11, 3))
plt.title('Good Life Camden Topics')
plt.barh(glc_grouped['glc_topics_cleansed'], glc_grouped['Count'], height=0.4, color='teal', alpha=0.5)
plt.yticks(fontsize=8)
plt.margins(y=0.05)
plt.show()


# ## Checking and examples of consolidated happiness and anxiety topic data

# In[ ]:


pd.set_option('display.max_rows', 500)
anx_top_check_df.sample(500)


# In[ ]:


path = 'E:/twitter_analysis_data/categorical_data_review/hap_anx_top_groupby_counts/'
anx_top_check_df.to_csv(path+ "anx_top_check_df.csv", index=False)


# In[ ]:


pd.set_option('display.max_rows', 500)
anx_top_check_df[anx_top_check_df['anx_topics_cleansed'].isin(['social','employment','politics', 'health'])]


# In[ ]:


pd.set_option('display.max_rows', 500)
hap_top_check_df.head(500)


# In[ ]:


pd.set_option('display.max_rows', 500)
hap_top_example_list = ['community connection', 'social interaction','sport','community events','design','self care','family connection','wellbeing']
hap_top_check_df[hap_top_check_df['hap_topics_cleansed'].isin(hap_top_example_list)]


# ## Term Similarity Analysis (run when required)
# ---

# In[ ]:


### DEFINE SIMILARITY FUNCTIONS
## LEVENSHTEIN DISTANCE (EDIT DISTANCE)
# define a function to return lev distance from a comparison between all items in a list of strings
def return_levenshtein_dist_df(input_list):
    lev_sims = []
    for i, s1 in enumerate(input_list):
         for j, s2 in enumerate(input_list):
            if i < j:  # avoid duplicate comparisons
                dist = levenshtein_distance(s1, s2)
                similarity = 1 - dist / max(len(s1), len(s2))
                similarity = round(similarity, 2)
                lev_sims.append((s1, s2, similarity))
    cols = ["term_1", "term_2", "lev_similarity"]
    lev_sim_df = pd.DataFrame(lev_sims, columns=cols)
    lev_sim_df = lev_sim_df.sort_values(by="lev_similarity", ascending=False)
    lev_sim_df = lev_sim_df.reset_index(drop=True)
    lev_sim_df['term_pair_key'] = lev_sim_df['term_1'] + "|" + lev_sim_df['term_2']
    col_order = ['term_pair_key', 'term_1', 'term_2', 'lev_similarity']
    lev_sim_df = lev_sim_df[col_order]
    return lev_sim_df

## COSINE SIMILARITY
# define a function to return cosine similarity from a comparison between all items in a list of strings
def return_cosine_similarity_df(input_list):
    # Convert strings to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(input_list)
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    # Create a DataFrame with one row per term pair combination
    similarities = []
    for i, term1 in enumerate(input_list):
        for j, term2 in enumerate(input_list):
            if i < j:  # Avoid duplicate pairs and self-comparisons
                similarities.append((term1, term2, similarity_matrix[i, j]))
    # Create the resulting DataFrame
    cosine_sim_df = pd.DataFrame(similarities, columns=["term_1", "term_2", "cosine_similarity"])
    cosine_sim_df = cosine_sim_df.sort_values(by='cosine_similarity', ascending=False)
    cosine_sim_df = cosine_sim_df.reset_index(drop=True)
    cosine_sim_df['cosine_similarity'] = [round(x,2) for x in cosine_sim_df['cosine_similarity']]
    cosine_sim_df['term_pair_key'] = cosine_sim_df['term_1'] + "|" + cosine_sim_df['term_2']
    col_order = ['term_pair_key', 'term_1', 'term_2', 'cosine_similarity']
    cosine_sim_df = cosine_sim_df[col_order]
    return cosine_sim_df

## JACCARD SIMILARITY
# define a function to return Jaccard similarity from a comparison between all items in a list of strings
def return_jaccard_similarity_df(input_list):
    def jaccard_similarity(s1, s2):
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / len(set1 | set2)
    similarities = []
    for i, s1 in enumerate(input_list):
        for j, s2 in enumerate(input_list):
            if i < j:
                similarity = jaccard_similarity(s1, s2)
                similarity = round(similarity, 2)
                similarities.append((s1, s2, similarity))   
    jaccard_sim_df = pd.DataFrame(similarities, columns=["term_1", "term_2", "jaccard_similarity"])
    jaccard_sim_df = jaccard_sim_df.sort_values(by='jaccard_similarity', ascending=False)
    jaccard_sim_df = jaccard_sim_df.reset_index(drop=True)
    jaccard_sim_df['term_pair_key'] = jaccard_sim_df['term_1'] + "|" + jaccard_sim_df['term_2']
    col_order = ['term_pair_key', 'term_1', 'term_2', 'jaccard_similarity']
    jaccard_sim_df = jaccard_sim_df[col_order]
    return jaccard_sim_df

## FUZZY WUZZY
# define a function to return Fuzz similarity from a comparison between all items in a list of strings
def return_fuzz_similarity_df(input_list):
    similarities = []
    for i, s1 in enumerate(input_list):
        for j, s2 in enumerate(input_list):
            if i < j:
                similarity = fuzz.ratio(s1, s2)
                similarity = similarity/100
                similarities.append((s1, s2, similarity))
    fuzz_sim_df = pd.DataFrame(similarities, columns=["term_1", "term_2", "fuzz_similarity"])
    fuzz_sim_df = fuzz_sim_df.sort_values(by='fuzz_similarity', ascending=False)
    fuzz_sim_df = fuzz_sim_df.reset_index(drop=True)
    fuzz_sim_df['term_pair_key'] = fuzz_sim_df['term_1'] + "|" + fuzz_sim_df['term_2']
    col_order = ['term_pair_key', 'term_1', 'term_2', 'fuzz_similarity']
    fuzz_sim_df = fuzz_sim_df[col_order]
    return fuzz_sim_df


# In[ ]:


### DEFINE A FUNCTION THAT RETURNS A DF FROM A LIST OF UNIQUE STRINGS APPLYING ALL 4 SIMILARITY MEASURES PLUS A COMPOSITE
def return_similarity_df(input_list):
    levevshtein_df = return_levenshtein_dist_df(input_list)
    print("Leveshtein difference computation completed")
    cosine_df = return_cosine_similarity_df(input_list)
    print("Cosine similarity computation completed")
    similarity_df = levevshtein_df.merge(cosine_df[['term_pair_key', 'cosine_similarity']], on="term_pair_key", how="left")
    del cosine_df
    gc.collect()
    jaccard_df = return_jaccard_similarity_df(input_list)
    print("Jaccard similarity computation completed")
    similarity_df = similarity_df.merge(jaccard_df[['term_pair_key', 'jaccard_similarity']], on="term_pair_key", how="left")
    del jaccard_df
    gc.collect()
    fuzz_df = return_fuzz_similarity_df(input_list)
    print("Fuzz similarity computation completed")
    similarity_df = similarity_df.merge(fuzz_df[['term_pair_key', 'fuzz_similarity']], on="term_pair_key", how="left")
    del fuzz_df
    gc.collect()
    similarity_df['composite_similarity'] = (similarity_df['lev_similarity']
                                             + similarity_df['cosine_similarity']
                                             + similarity_df['jaccard_similarity']
                                             + similarity_df['fuzz_similarity']) / 4
    similarity_df['composite_similarity'] = [round(x,2) for x in similarity_df['composite_similarity']]
    similarity_df = similarity_df.sort_values(by='composite_similarity', ascending=False)
    return similarity_df


# In[ ]:


############### LONG RUN: CALCULATE AND STORE HAP SIMILARITY SCORES ##################
path = "E:/twitter_analysis_data/categorical_data_review/similarity scores/"
hap_top_sim_df = return_similarity_df(unique_hap_top_values)
hap_top_sim_df = hap_top_sim_df[hap_top_sim_df['composite_similarity'] > 0.5]
hap_top_sim_df.to_parquet(path + "hap_top_sim_df_parquet.parquet", engine="pyarrow")
del hap_top_sim_df
gc.collect()


# In[ ]:


############### LONG RUN: CALCULATE AND STORE ANX SIMILARITY SCORES ##################
path = "E:/twitter_analysis_data/categorical_data_review/similarity scores/"
anx_top_sim_df = return_similarity_df(unique_anx_top_values)
anx_top_sim_df = anx_top_sim_df[anx_top_sim_df['composite_similarity'] > 0.5]
anx_top_sim_df.to_parquet(path + "anx_top_sim_df_parquet.parquet", engine="pyarrow")
del anx_top_sim_df
gc.collect()


# In[ ]:


############### LONG RUN: CALCULATE AND STORE EMOTION SIMILARITY SCORES ##################
path = "E:/twitter_analysis_data/categorical_data_review/similarity scores/"
emo_sim_df = return_similarity_df(unique_emo_values)
emo_sim_df = emo_sim_df[emo_sim_df['composite_similarity'] > 0]
emo_sim_df.to_parquet(path + "emo_sim_df_parquet.parquet", engine="pyarrow")
del emo_sim_df
gc.collect()


# In[ ]:


### MANUAL IDENTIFICATION OF ASSOCIATED CATEGORICAL VALUES FOR CONSOLIDATION


# In[ ]:


# load parquet files of similarity calculations
path = "E:/twitter_analysis_data/categorical_data_review/similarity scores/"
anx_top_sim_df = pd.read_parquet(path + 'anx_top_sim_df_parquet.parquet')
hap_top_sim_df = pd.read_parquet(path + 'hap_top_sim_df_parquet.parquet')
#emo_sim_df = pd.read_parquet(path + 'emo_sim_df_parquet.parquet')


# In[ ]:





# In[ ]:


#remove numbers
# replace "for" with "of"


# In[ ]:


anx_top_sim_df.sort_values(by="composite_similarity", ascending=False).head(500)


# In[ ]:


hap_top_sim_df.sort_values(by="composite_similarity", ascending=False).head(500)


# In[ ]:


hap_top_sim_df.to_csv(path + "hap_top_sim_df_csv.csv", index=False)
anx_top_sim_df.to_csv(path + "hap_top_sim_df_csv.csv", index=False)


# In[ ]:


hap_top_sim_df.sample(3)


# In[ ]:


emo_sim_df.sample(3)


# In[ ]:


#del anx_top_sim_df
del hap_top_sim_df
#del emo_sim_df
gc.collect()


# In[ ]:


emo_sim_df = emo_sim_df.sort_values(by='composite_similarity', ascending=False)


# In[ ]:


emo_sim_df[emo_sim_df['composite_similarity'] > 0.5].sample(50).sort_values(by='composite_similarity', ascending=False)


# In[ ]:


plt.hist(anx_top_sim_df['composite_similarity'], bins=100, edgecolor='black')
plt.show()


# ## Model cleansed categorical data back into main DF
# ---

# In[ ]:


# create new dataframe squishing data previously exploded for term consolidation
lang_df_squish = lang_df.groupby('concat_key')['languages_cleansed'].apply(list).reset_index()
hap_top_df_squish = hap_top_df.groupby('concat_key')['hap_topics_cleansed'].apply(list).reset_index()
anx_top_df_squish = anx_top_df.groupby('concat_key')['anx_topics_cleansed'].apply(list).reset_index()
emo_df_squish = emo_df.groupby('concat_key')['emotions_cleansed'].apply(list).reset_index()
glc_df_squish = glc_df.groupby('concat_key')['glc_topics_cleansed'].apply(list).reset_index()
# combine squished data into main dataframe
cleansed_dfs = [lang_df_squish, hap_top_df_squish, anx_top_df_squish, emo_df_squish, glc_df_squish]
for c_df in cleansed_dfs:
    df = df.merge(c_df, on='concat_key', how='left')
df['languages_cleansed'] = df['languages_cleansed'].apply(lambda x: x if isinstance(x, list) else [])
df['hap_topics_cleansed'] = df['hap_topics_cleansed'].apply(lambda x: x if isinstance(x, list) else [])
df['anx_topics_cleansed'] = df['anx_topics_cleansed'].apply(lambda x: x if isinstance(x, list) else [])
df['emotions_cleansed'] = df['emotions_cleansed'].apply(lambda x: x if isinstance(x, list) else [])
df['glc_topics_cleansed'] = df['glc_topics_cleansed'].apply(lambda x: x if isinstance(x, list) else [])


# # CROSS REFERENCING ONS PERSONAL WELLBEING DATA
# ---

# ## Agregating to financial year and visualising against ONS Personal Wellbeing

# In[ ]:


### CREATE MEAN HAP ANX LVL YEAR-ON-YEAR COMPARISON DATA

# create financial year mean values df
fin_year_means = df.groupby("financial_year").agg(
    mean_hap_lvl=("hap_lvl", lambda x: round(x.mean(), 2)),
    mean_anx_lvl=("anx_lvl", lambda x: round(x.mean(), 2))
).reset_index()

# load ONS mean values by financial year
ons_means = {"financial_year": ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"],
    "ons_cam_mean_hap": [7.11, 7.38, 7.05, 6.92, 7.16, 7.28],"ons_cam_mean_anx": [3.62, 3.3, 3.47, 3.64, 3.73, 3.37]}
ons_means = pd.DataFrame(ons_means)

# merge dataframes
merged_df = ons_means.merge(fin_year_means, on="financial_year")

# Creating the figure with an adjusted layout
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1])
ax_top = fig.add_subplot(gs[0, :])  # Top visualization spans full width
ax_hap = fig.add_subplot(gs[1, 0])  # Bottom-left
ax_anx = fig.add_subplot(gs[1, 1])  # Bottom-right

# Full data visualization (Top Chart)
ax_top.plot(merged_df["financial_year"], merged_df["ons_cam_mean_hap"], color="#FFC000", linestyle="-", marker='o', label="ONS Happiness")
ax_top.plot(merged_df["financial_year"], merged_df["mean_hap_lvl"], color="#FFC000", linestyle="--", marker='x', label="Twitter Happiness")
ax_top.plot(merged_df["financial_year"], merged_df["ons_cam_mean_anx"], color="grey", linestyle="-", marker='o', label="ONS Anxiety")
ax_top.plot(merged_df["financial_year"], merged_df["mean_anx_lvl"], color="grey", linestyle="--", marker='x', label="Twitter Anxiety")
ax_top.set_ylim(0, 10)
ax_top.tick_params(axis='y', labelsize=14)
ax_top.tick_params(axis='x', labelsize=14)
#ax_top.set_xlabel("Financial Year")
#ax_top.set_ylabel("Well-being Score")
ax_top.set_title("Personal Well-being in Camden (ONS / Twitter Comparison)", loc='left', fontsize=18, pad=10)
ax_top.legend(fontsize="12", loc ="upper right")
ax_top.grid(True)

# Mean Happiness Levels (Bottom-left)
ax_hap.plot(merged_df["financial_year"], merged_df["ons_cam_mean_hap"], color="#FFC000", linestyle="-", marker='o', label="ONS Happiness")
ax_hap.plot(merged_df["financial_year"], merged_df["mean_hap_lvl"], color="#FFC000", linestyle="--", marker='x', label="Local Happiness")
#ax_hap.set_xlabel("Financial Year")
#ax_hap.set_ylabel("Happiness Level")
ax_hap.set_title("Average Happiness Levels", loc='left', color="#FFC000", fontweight="bold")
#ax_hap.legend()
ax_hap.tick_params(axis='y', colors="#FFC000")
plt.setp(ax_hap.get_yticklabels(), fontweight="bold")
ax_hap.grid(True)

# Mean Anxiety Levels (Bottom-right)
ax_anx.plot(merged_df["financial_year"], merged_df["ons_cam_mean_anx"], color="grey", linestyle="-", marker='o', label="ONS Anxiety")
ax_anx.plot(merged_df["financial_year"], merged_df["mean_anx_lvl"], color="grey", linestyle="--", marker='x', label="Local Anxiety")
#ax_anx.set_xlabel("Financial Year")
#ax_anx.set_ylabel("Anxiety Level")
ax_anx.tick_params(axis='y', colors="grey")
plt.setp(ax_anx.get_yticklabels(), fontweight="bold")
ax_anx.set_title("Average Anxiety Levels", loc='left', color="grey", fontweight="bold")
#ax_anx.legend()
ax_anx.grid(True)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()


# # LONGITUDINAL ANALYSIS OF HAPPINESS AND ANXIETY LEVELS
# ---

# ## Change in mean happiness and anxiety month on month

# In[ ]:


# create new df filtering to columns required for analysis
hap_anx_longitudinal_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text',
                            'hap_lvl', 'anx_lvl', 'hap_topics_cleansed', 'anx_topics_cleansed']
hap_anx_longitudinal_df = df[hap_anx_longitudinal_cols].copy()


# In[ ]:


mean_hap_anx_mnth = hap_anx_longitudinal_df.groupby('mnth_yr_text').agg({'hap_lvl': 'mean', 'anx_lvl': 'mean'}).reset_index()
# Convert 'mnth_yr_text' to datetime format
mean_hap_anx_mnth['date'] = pd.to_datetime(mean_hap_anx_mnth['mnth_yr_text'], format='%B %Y')
# Sort DataFrame by the date
mean_hap_anx_mnth = mean_hap_anx_mnth.sort_values(by='date').reset_index(drop=True)


# In[ ]:


# Define period markers for February 2020 to January 2021
start_period = pd.to_datetime('February 2020')
end_period = pd.to_datetime('January 2021')

## Apply smoothing
# Create a function for smoothing the lines
def smooth_line(x, y, num_points=300):
    """Interpolates data points to create a smooth curve."""
    x_numeric = np.array(x.map(pd.Timestamp.toordinal))  # Convert datetime to ordinal values
    y_numeric = np.array(y)  
    # Create an interpolation function
    spline = make_interp_spline(x_numeric, y_numeric, k=3)  # k=3 for cubic spline
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), num_points)
    y_smooth = spline(x_smooth)
    # Convert x_smooth back to datetime
    x_smooth_dates = [pd.Timestamp.fromordinal(int(val)) for val in x_smooth]
    return x_smooth_dates, y_smooth
# Generate smoothed data for hap_lvl
x_smooth_hap, y_smooth_hap = smooth_line(mean_hap_anx_mnth['date'], mean_hap_anx_mnth['hap_lvl'])
# Generate smoothed data for anx_lvl
x_smooth_anx, y_smooth_anx = smooth_line(mean_hap_anx_mnth['date'], mean_hap_anx_mnth['anx_lvl'])

# Plot Happiness Level Trend
plt.figure(figsize=(13, 5))
plt.axvspan(start_period, end_period, color='#51a3a3', alpha=0.1)  # Highlight period with a shaded rectangle
plt.plot(x_smooth_hap, y_smooth_hap, color='#FFC000', linestyle='-', linewidth=2)
#plt.xlabel('Date')
#plt.ylabel('Happiness Level')
plt.title('Average Happiness Level Month-on-month', loc='left', fontsize=13, pad=8, color="#FFC000", fontweight="bold")
plt.xticks(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot Anxiety Level Trend
plt.figure(figsize=(13, 5))
plt.axvspan(start_period, end_period, color='#51a3a3', alpha=0.1)  # Highlight period with a shaded rectangle
plt.plot(x_smooth_anx, y_smooth_anx, color='grey', linestyle='-', linewidth=2)
#plt.xlabel('Date')
#plt.ylabel('Anxiety Level')
plt.title('Average Anxiety Level Month-on-month', loc='left', fontsize=13, pad=8, color="grey", fontweight="bold")
plt.xticks(fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# Staistical summary of high level trend and calculation of statistical signifcance

# # DRIVERS OF HAPPINESS AND ANXIETY
# ---

# ## Simple Word Clouds

# In[ ]:


## create count dictionaries from which to build WordColouds
# define a function to deriver word count dict
def unique_item_occurence_counts(series):
    list_of_lists = []
    for x in series:    
        for item in x:
            list_of_lists.append(item)        
    counts_dict = Counter(list_of_lists)
    return counts_dict
# create count dicts
hap_topic_count_dict = unique_item_occurence_counts(df['hap_topics_cleansed'])
anx_topic_count_dict = unique_item_occurence_counts(df['anx_topics_cleansed'])


# In[ ]:


## VISUALISE HAPPINESS AND ANXIETY TOPICS AS SIMPLE WORDCLOUDS (with random colour shading)
# create WordCloud objects for both dictionaries
hap_wc = WordCloud(
    width=1000,
    height=300,
    background_color='white',
    colormap='Wistia',
    prefer_horizontal=0.9,
    scale=10,
    random_state=2,
).generate_from_frequencies(hap_topic_count_dict)
anx_wc = WordCloud(
    width=1000,
    height=300,
    background_color='white',
    colormap='gray',
    prefer_horizontal=0.9,
    scale=10,
    random_state=2,
).generate_from_frequencies(anx_topic_count_dict)
# plot the word clouds side by side
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
# hap word cloud
axes[0].imshow(hap_wc, interpolation='bilinear')
axes[0].set_title("Topics contributing to Happiness:", fontsize=9, loc='left', color='black', pad=10)
axes[0].axis('off')
# anx word cloud
axes[1].imshow(anx_wc, interpolation='bilinear')
axes[1].set_title("Topics contributing to Anxiety:", fontsize=9, loc='left', color='black', pad=10)
axes[1].axis('off')
# display the plot
plt.tight_layout()
plt.show()


# In[ ]:


## VISUALISE HAPPINESS AND ANXIETY TOPICS AS SINGLE WORDCLOUD WITH CAMDEN MASK

# process words to visualize overlaps separately
combined_word_counts = {}
hap_topic_words = set(hap_topic_count_dict.keys())
anx_topic_words = set(anx_topic_count_dict.keys())
overlapping_words = hap_topic_words & anx_topic_words

# create combined dictionary (with overlapping words delineated with a " " to indicate and category) 
for word, freq in hap_topic_count_dict.items():
        combined_word_counts[word] = freq
for word, freq in anx_topic_count_dict.items():
    if word in overlapping_words:
        combined_word_counts[word + " "] = freq
    else:
        combined_word_counts[word] = freq

# initialise sets and manage overlaps
hap_words_set = set(hap_topic_count_dict.keys())
anx_words_set = set(anx_topic_count_dict.keys())
overlap_words_set = hap_words_set.intersection(anx_words_set)
anx_words_minus_overlap_set = anx_words_set.symmetric_difference(hap_words_set)
overlap_with_spcae_set = {s + " " for s in overlap_words_set}
anx_words_set_viz = anx_words_minus_overlap_set | overlap_with_spcae_set
hap_words_set_viz = hap_words_set

# define class objects
class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping
       Parameters
        - color_to_words : dict(str -> list(str))   A dictionary that maps a color to the list of words.
        - default_color : str   Color that will be assigned to a word that's not a member of any value from color_to_words.
    """
    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color
    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)
class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.
       Uses wordcloud.get_single_color_func
       Parameters: 
           - color_to_words : dict(str -> list(str))    A dictionary that maps a color to the list of words.
           - default_color : str    Color that will be assigned to a word that's not a member of any value from color_to_words.
    """
    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]
        self.default_color_func = get_single_color_func(default_color)
    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func
        return color_func
    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

# load Camden mask
camden_mask = np.array(Image.open("C:/Users/Edward/Documents/Twitter Analysis Notebooks/suplimentary_files/camden_mask.png"))

# intitalise wc word cloud object dpcs:https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
wc = WordCloud(
    width=500,
    height=500,
    background_color='white',
    mode='RGBA',
    mask=camden_mask,
    prefer_horizontal=0.8,
    relative_scaling=0.,
    #random_state=1,
).generate_from_frequencies(combined_word_counts)

# colour to words dicts
color_to_words = {
    # words below will be colored with a green single color function
    '#FFC000': list(hap_words_set_viz),
    # will be colored with a red single color function
    'grey': list(anx_words_set_viz)
}
# words that are not in any of the color_to_words values  will be colored with a grey single color function
default_color = 'grey'
# Create a color function with single tone
# grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
# Create a color function with multiple tones
grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
# Apply our color function
wc.recolor(color_func=grouped_color_func)

# save image
path = 'E:/twitter_analysis_data/viz/lbc_wc/'
x = random.randint(10000000, 99999999)
wc.to_file(path + str(x) + "wordcloud.png")  # This may not support transparency in some versions
plt.savefig(path + str(x) + "_lcb_hap_anx_wc_trans.png", transparent=True, format="png", dpi=160, bbox_inches='tight')

# plot visualisation
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# ## TF-IDF for 3 chronoclogial Periods (experimental)

# In[ ]:


# create new df filtering to columns required for analysis
hap_anx_longitudinal_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text',
                            'hap_lvl', 'anx_lvl', 'hap_topics_cleansed', 'anx_topics_cleansed']
hap_anx_longitudinal_df = df[hap_anx_longitudinal_cols].copy()


# In[ ]:


hap_anx_longitudinal_df.sample(3)


# In[ ]:


## DELINEATING 3 PERIODS
# convert 'mnth_yr_text' to datetime format
hap_anx_longitudinal_df['date'] = pd.to_datetime(df['mnth_yr_text'], format='%B %Y')
# define category labels based on the given periods
def categorise_period(date):
    if date >= pd.to_datetime('January 2018') and date <= pd.to_datetime('January 2020'):
        return 'Jan 2018 - Jan 2020'
    elif date >= pd.to_datetime('February 2020') and date <= pd.to_datetime('January 2021'):
        return 'Feb 2020 - Jan 2021'
    elif date >= pd.to_datetime('February 2021') and date <= pd.to_datetime('March 2023'):
        return 'Feb 2021 - Mar 2023'
    else:
        return 'Other'  # In case there are unexpected values
# apply the categoriSation function
hap_anx_longitudinal_df['period_category'] = hap_anx_longitudinal_df['date'].apply(categorise_period)


# In[ ]:





# In[ ]:


hap_anx_longitudinal_df.sample(3)


# In[ ]:


hap_anx_longitudinal_df.columns


# In[ ]:


hap_anx_longitudinal_df.shape


# In[ ]:


### REMODELING TOPIC DATA FOR TF-IDF COMPUTATION ###


# In[ ]:


## create corpus dict for happiness topics
# Filter out rows where 'hap_topics_cleansed' is empty
temp_tfidf_df_hap = hap_anx_longitudinal_df[
    hap_anx_longitudinal_df['hap_topics_cleansed'].apply(lambda x: isinstance(x, list) and len(x) > 0)
].copy()
# Explode lists so each element gets its own row
temp_tfidf_df_hap = temp_tfidf_df_hap.explode('hap_topics_cleansed')
# Ensure 'hap_topics_cleansed' contains only string values
temp_tfidf_df_hap['hap_topics_cleansed'] = temp_tfidf_df_hap['hap_topics_cleansed'].astype(str)
# Remodel topic data into a dictionary (key: period_category, value: list of strings)
hap_period_dict = temp_tfidf_df_hap.groupby('period_category')['hap_topics_cleansed'].agg(list).to_dict()
# specify the desired order
period_order = ['Jan 2018 - Jan 2020', 'Feb 2020 - Jan 2021', 'Feb 2021 - Mar 2023']
# Reconstruct dictionary in the specified order
hap_period_dict = {key: hap_period_dict[key] for key in period_order}
# de,ete temp df
del temp_tfidf_df_hap

## create corpus dict for anxiety topics
# Filter out rows where 'anx_topics_cleansed' is empty
temp_tfidf_df_anx = hap_anx_longitudinal_df[
    hap_anx_longitudinal_df['anx_topics_cleansed'].apply(lambda x: isinstance(x, list) and len(x) > 0)
].copy()
# Explode lists so each element gets its own row
temp_tfidf_df_anx = temp_tfidf_df_anx.explode('anx_topics_cleansed')
# Ensure 'anx_topics_cleansed' contains only string values
temp_tfidf_df_anx['anx_topics_cleansed'] = temp_tfidf_df_anx['anx_topics_cleansed'].astype(str)
# Remodel topic data into a dictionary (key: period_category, value: list of strings)
anx_period_dict = temp_tfidf_df_anx.groupby('period_category')['anx_topics_cleansed'].agg(list).to_dict()
# specify the desired order
period_order = ['Jan 2018 - Jan 2020', 'Feb 2020 - Jan 2021', 'Feb 2021 - Mar 2023']
# Reconstruct dictionary in the specified order
anx_period_dict = {key: anx_period_dict[key] for key in period_order}
# de,ete temp df
del temp_tfidf_df_anx


# In[ ]:


### output dicts as json
period_tfidf_path = 'E:/twitter_analysis_data/categorical_data_review/tfidf 3 period analysis/'
with open(period_tfidf_path + "hap_3_doc_dict.json", "w") as json_file:
    json.dump(hap_period_dict, json_file, indent=4)
with open(period_tfidf_path + "anx_3_doc_dict.json", "w") as json_file:
    json.dump(anx_period_dict, json_file, indent=4)


# In[ ]:


hap_period_dict


# In[ ]:


anx_period_dict


# In[ ]:


### ANX TFIDF ANALYSIS



# ![image.png](attachment:c6572c20-b8f9-4765-9e80-c806ded22c8b.png)

# In[ ]:


# Number of documents
N = len(anx_period_dict)

# Step 1: Calculate Term Frequency (TF) for each document
tf_documents = {doc: Counter(terms) for doc, terms in anx_period_dict.items()}

# Step 2: Compute Document Frequency (DF)
doc_freq = Counter()
for terms in tf_documents.values():
    for term in terms.keys():
        doc_freq[term] += 1

# Step 3: Compute Inverse Document Frequency (IDF)
inv_doc_freq = {term: math.log(N / doc_freq[term]) + 1 for term in doc_freq}

# Step 4: Compute TF-IDF for each document
tf_idf_documents = {
    doc: {term: tf * inv_doc_freq[term] for term, tf in tf_counter.items()}
    for doc, tf_counter in tf_documents.items()
}

# Convert to a DataFrame for better visualization
tf_idf_df = pd.DataFrame(tf_idf_documents).fillna(0)


# In[ ]:


tf_idf_df


# In[ ]:


### DEGING TF-IDF FUNCTION ###


# In[ ]:





# In[ ]:


### APPLYING TFIDF FUNCTION ###


# In[ ]:





# In[ ]:





# In[ ]:





# ## Longitudinal Analysis of Thematic Drivers of Anxiety and Happiness
# ---
# ?mean values after filtering? ? prevalence of topic string?
# nb: retain visualisation of mean across full data for reference

# In[ ]:


# create new df filtering to columns required for analysis
hap_anx_longitudinal_cols = ['concat_key', 'user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text',
                            'hap_lvl', 'anx_lvl', 'hap_topics_cleansed', 'anx_topics_cleansed']
hap_anx_longitudinal_df = df[hap_anx_longitudinal_cols].copy()
hap_anx_longitudinal_df['date'] = pd.to_datetime(df['mnth_yr_text'], format='%B %Y')
#create exploded hap df
hap_longitudinal_df = hap_anx_longitudinal_df.explode('hap_topics_cleansed', ignore_index=True).dropna(subset=['hap_topics_cleansed'])
hap_longitudinal_df.drop(['anx_topics_cleansed', 'anx_lvl'], axis=1, inplace=True)
#create exploded anx df
anx_longitudinal_df = hap_anx_longitudinal_df.explode('anx_topics_cleansed', ignore_index=True).dropna(subset=['anx_topics_cleansed'])
anx_longitudinal_df.drop(['hap_topics_cleansed', 'hap_lvl'], axis=1, inplace=True)


# Cleansed topic data were categorised into themes via a combination of manual review and categorisation and few-shot learnbing using GPT4o mini.

# In[ ]:


#del anx_themes, hap_themes, hap_anx_longitudinal_df, hap_longitudinal_df, anx_longitudinal_df


# In[ ]:


themes_path = 'E:/twitter_analysis_data/categorical_data_review/hap_anx_top_groupby_counts/labeled terms/final_theme_dfs/'
# load in term/theme datasets (managingh formating issues)
anx_themes = pd.read_csv(themes_path + 'anx_themes.csv', encoding='ISO-8859-1')
anx_themes = anx_themes.rename(columns={'term': 'anx_topics_cleansed'})
hap_themes = pd.read_csv(themes_path + 'hap_themes.csv')
hap_themes = hap_themes.rename(columns={'term': 'hap_topics_cleansed'})
#---
# left join themes into longitudinal datasets
anx_longitudinal_df = anx_longitudinal_df.merge(anx_themes, on='anx_topics_cleansed', how='left')
hap_longitudinal_df = hap_longitudinal_df.merge(hap_themes, on='hap_topics_cleansed', how='left')
#---
# drop rows with no themes
anx_longitudinal_df = anx_longitudinal_df.dropna(subset=['theme'])
hap_longitudinal_df = hap_longitudinal_df.dropna(subset=['theme'])
# ---
anx_longitudinal_df.loc[:, 'theme'] = [s.title() for s in anx_longitudinal_df['theme']]
anx_longitudinal_df.loc[:, 'theme'] = [s.replace('And','and') for s in anx_longitudinal_df['theme']]
hap_longitudinal_df.loc[:, 'theme'] = [s.title() for s in hap_longitudinal_df['theme']]
hap_longitudinal_df.loc[:, 'theme'] = [s.replace('And','and') for s in hap_longitudinal_df['theme']]
anx_longitudinal_df.loc[:, 'theme'] = anx_longitudinal_df['theme'].str.strip()
hap_longitudinal_df.loc[:, 'theme'] = hap_longitudinal_df['theme'].str.strip()
anx_longitudinal_df.loc[:, 'theme'] = [s.replace('Finances and Employment','Personal Finances and Employment') for s in anx_longitudinal_df['theme']]


# In[ ]:


### MERGE HAPPINESS - WEATHER AND SEASONAL WITH CELEBRATION ---
hap_longitudinal_df.loc[:, 'theme'] = [s.replace('Celebration','Celebration and Seasonal') for s in hap_longitudinal_df['theme']]
hap_longitudinal_df.loc[:, 'theme'] = [s.replace('Weather and Seasonal','Celebration and Seasonal') for s in hap_longitudinal_df['theme']]


# In[ ]:


hap_longitudinal_df['theme'].unique()


# In[ ]:


hap_longitudinal_df['theme'].unique()


# In[ ]:


anx_longitudinal_df


# In[ ]:


## checking


# In[ ]:


print("anx_long_df shape: " + str(anx_longitudinal_df.shape))
anx_longitudinal_df.sample(3)


# In[ ]:


print("anx_long_df shape: " + str(hap_longitudinal_df.shape))
hap_longitudinal_df.sample(3)


# Thematic Analysis

# In[ ]:


### GROUP BY COUNTS
# hap topic groupby counts
hap_theme_groupby_counts = hap_longitudinal_df.groupby('theme')['theme'].count()
hap_theme_groupby_counts = pd.DataFrame(hap_theme_groupby_counts)
hap_theme_groupby_counts = hap_theme_groupby_counts.rename(columns={'theme':'count'})
hap_theme_groupby_counts = hap_theme_groupby_counts.sort_values(by='count', ascending=False)
hap_theme_groupby_counts = hap_theme_groupby_counts.reset_index()
# hap topic groupby counts
anx_theme_groupby_counts = anx_longitudinal_df.groupby('theme')['theme'].count()
anx_theme_groupby_counts = pd.DataFrame(anx_theme_groupby_counts)
anx_theme_groupby_counts = anx_theme_groupby_counts.rename(columns={'theme':'count'})
anx_theme_groupby_counts = anx_theme_groupby_counts.sort_values(by='count', ascending=False)
anx_theme_groupby_counts = anx_theme_groupby_counts.reset_index()


# In[ ]:


anx_theme_groupby_counts


# In[ ]:


anx_theme_groupby_counts


# In[ ]:


hap_theme_groupby_counts


# In[ ]:


## save 
theme_analysis_path = 'E:/twitter_analysis_data/categorical_data_review/theme_level_analysis/'
anx_longitudinal_df.to_csv(theme_analysis_path + 'anx_longitudinal_df.csv', index=False)
hap_longitudinal_df.to_csv(theme_analysis_path + 'hap_longitudinal_df.csv', index=False)


# In[ ]:


###### ANXIETY LINE VIZ #######


# In[ ]:


## CREATE PIVOT OF % MEASURE (% of users in month with topic under theme identified)
anx_df = anx_longitudinal_df
# Convert 'date' to datetime format
anx_df['date'] = pd.to_datetime(anx_df['date'])
# Extract year-month for grouping
anx_df['year_month'] = anx_df['date'].dt.to_period('M')
# Group by theme and year-month, count distinct 'concat_key' values
anx_theme_month_counts = anx_df.groupby(["theme", "year_month"])["concat_key"].nunique().reset_index()
# Get the total distinct 'concat_key' count per month
anx_total_month_counts = anx_df.groupby("year_month")["concat_key"].nunique().reset_index()
anx_total_month_counts = anx_total_month_counts.rename(columns={"concat_key": "total_count"})
# Merge the counts
anx_theme_month_counts = anx_theme_month_counts.merge(anx_total_month_counts, on="year_month")
# Calculate percentage
anx_theme_month_counts["percentage"] = (anx_theme_month_counts["concat_key"] / anx_theme_month_counts["total_count"]) * 100
# Pivot the table with columns in chronological order
anx_pivot_table = anx_theme_month_counts.pivot(index="theme", columns="year_month", values="percentage").fillna(0)
anx_pivot_table = anx_pivot_table.round(2)
anx_pivot_table = anx_pivot_table.sort_index(axis=1)  # Ensure chronological order
anx_pivot_table.index = [s.title() for s in anx_pivot_table.index]
anx_pivot_table.index = [s.replace("And","and") for s in anx_pivot_table.index]
anx_pivot_table = anx_pivot_table.round(2)
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/'
anx_pivot_table.to_csv(save_path + 'anx_pivot_table.csv', index=True)
print("% of active users with anxiety driver theme identified per month:")
anx_pivot_table


# In[ ]:


# Plot each theme as a separate line chart stacked vertically
themes = anx_pivot_table.index
#themes = [s.title() for s in themes]
fig, axes = plt.subplots(len(themes), 1, figsize=(12, 3 * len(themes)), sharex=True)
for ax, theme in zip(axes, themes):
    ax.plot(anx_pivot_table.columns.astype(str), anx_pivot_table.loc[theme], marker='o', linestyle='-', color='grey', markerfacecolor='grey', markeredgecolor='grey')
    ax.set_title(theme, fontsize=15, loc='left', fontweight='bold', color='grey')  # Title aligned to the left
    ax.grid(True)
    ax.set_ylabel("")  # Remove the "Percentage" y-axis label
    ax.tick_params(axis='y', labelsize=12)
# Modify x-axis ticks to display only year values on the last chart
last_ax = axes[-1]
last_ax.set_xticks(range(0, len(anx_pivot_table.columns), 12))  # Show only yearly intervals
last_ax.set_xticklabels([str(col)[:4] for col in anx_pivot_table.columns[::12]], fontsize=13)  # Extract year only
#save figure
save_path = "E:/twitter_analysis_data/viz/theme_analysis/anx_themes.png"
plt.savefig(save_path, dpi=1200,pad_inches=0.1,bbox_inches='tight')
#plt.tight_layout()
plt.show()


# In[ ]:


##### HAPPINESS LINE VIZ #####


# In[ ]:


## CREATE PIVOT OF % MEASURE (% of users in month with topic under theme identified)
hap_df = hap_longitudinal_df
# Convert 'date' to datetime format
hap_df['date'] = pd.to_datetime(hap_df['date'])
# Extract year-month for grouping
hap_df['year_month'] = hap_df['date'].dt.to_period('M')
# Group by theme and year-month, count distinct 'concat_key' values
hap_theme_month_counts = hap_df.groupby(["theme", "year_month"])["concat_key"].nunique().reset_index()
# Get the total distinct 'concat_key' count per month
hap_total_month_counts = hap_df.groupby("year_month")["concat_key"].nunique().reset_index()
hap_total_month_counts = hap_total_month_counts.rename(columns={"concat_key": "total_count"})
# Merge the counts
hap_theme_month_counts = hap_theme_month_counts.merge(hap_total_month_counts, on="year_month")
# Calculate percentage
hap_theme_month_counts["percentage"] = (hap_theme_month_counts["concat_key"] / hap_theme_month_counts["total_count"]) * 100
# Pivot the table with columns in chronological order
hap_pivot_table = hap_theme_month_counts.pivot(index="theme", columns="year_month", values="percentage").fillna(0)
hap_pivot_table = hap_pivot_table.round(2)
hap_pivot_table = hap_pivot_table.sort_index(axis=1)  # Ensure chronological order
hap_pivot_table.index = [s.title() for s in hap_pivot_table.index]
hap_pivot_table.index = [s.replace("And","and") for s in hap_pivot_table.index]
hap_pivot_table = hap_pivot_table.round(2)
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/'
hap_pivot_table.to_csv(save_path + 'hap_pivot_table.csv', index=True)
print("% of active users with happiness driver theme identified per month:")
hap_pivot_table


# In[ ]:


# Plot each theme as a separate line chart stacked vertically
themes = hap_pivot_table.index
fig, axes = plt.subplots(len(themes), 1, figsize=(12, 3 * len(themes)), sharex=True)
for ax, theme in zip(axes, themes):
    ax.plot(hap_pivot_table.columns.astype(str), hap_pivot_table.loc[theme], marker='o', linestyle='-', color='#FFC000', markerfacecolor='#FFC000', markeredgecolor='#FFC000')
    ax.set_title(theme, fontsize=15, loc='left', fontweight='bold', color='#FFC000')  # Title aligned to the left
    ax.grid(True)
    ax.set_ylabel("")  # Remove the "Percentage" y-axis label
    ax.tick_params(axis='y', labelsize=12)
# Modify x-axis ticks to display only year values on the last chart
last_ax = axes[-1]
last_ax.set_xticks(range(0, len(hap_pivot_table.columns), 12))  # Show only yearly intervals
last_ax.set_xticklabels([str(col)[:4] for col in hap_pivot_table.columns[::12]], fontsize=13)  # Extract year only
#save figure
save_path = "E:/twitter_analysis_data/viz/theme_analysis/hap_themes.png"
plt.savefig(save_path, dpi=1200,pad_inches=0.1,bbox_inches='tight')
#plt.tight_layout()
plt.show()


# Exporing detail of themes that show particular temporal trend with reference to releveant events

# In[ ]:


### GENERATE DATASET FOR STATISTICAL SIGNIFICANCE TESTING
# a dataset that can be used to carry out the analysis of statistical significance of any trends identified with analysis
# of longitudinal trend in prevelance of a particular them 


# In[ ]:


### ANX STAT SIG DATASET


# In[ ]:


anx_longitudinal_df[anx_longitudinal_df['user_id'] == '972246044']


# ### Statistical Modeling Function
# ---

# In[ ]:


## create anxiety dataset for statistical mdoeling
anx_stat_sig_df = anx_longitudinal_df[['user_id', 'date', 'theme']]
# drop dupes
anx_stat_sig_df = anx_stat_sig_df.drop_duplicates(subset=['user_id', 'date', 'theme']).copy()
#create a column with value 'Y' for marking presence of theme
anx_stat_sig_df['value'] = 'Y'
#pivot the table
anx_stat_sig_df = anx_stat_sig_df.pivot_table(index=['user_id', 'date'], columns='theme', values='value', aggfunc='first', fill_value='N').reset_index()
anx_stat_sig_df.to_csv('E:/twitter_analysis_data/stat_sig_testing/anx_stat_sig.csv', index=False)


# In[ ]:


## create happiness dataset for statistical mdoeling
hap_stat_sig_df = hap_longitudinal_df[['user_id', 'date', 'theme']]
# drop dupes
hap_stat_sig_df = hap_stat_sig_df.drop_duplicates(subset=['user_id', 'date', 'theme']).copy()
#create a column with value 'Y' for marking presence of theme
hap_stat_sig_df['value'] = 'Y'
#pivot the table
hap_stat_sig_df = hap_stat_sig_df.pivot_table(index=['user_id', 'date'], columns='theme', values='value', aggfunc='first', fill_value='N').reset_index()
hap_stat_sig_df.to_csv('E:/twitter_analysis_data/stat_sig_testing/hap_stat_sig.csv', index=False)


# In[ ]:


def stat_analysis_for_theme(
    anx_or_hap,
    input_dataframe,
    theme,
    periods,
    logit_text_positions,
    gee_text_positions,
    line_color_data="grey",
    r_p_pos = (0.98, 0.95)
):
    print("\n\n\033[1mSTATISTICAL ANALYSIS FOR THEME:")
    
     # Convert date and map theme to binary
    df_local = input_dataframe.copy()
    df_local['date'] = pd.to_datetime(df_local['date'])
    df_local['theme_binary'] = df_local[theme].map({'Y': 1, 'N': 0})

    # Monthly prevalence
    monthly_series = df_local.groupby(df_local['date'].dt.to_period("M"))['theme_binary'].mean().reset_index()
    monthly_series['date'] = monthly_series['date'].dt.to_timestamp()

    
    # Permutation test
    print("\n\033[1mTESTING CUBIC SPLINE INTERPOLATION MODEL VERSUS NULL HYPOTHESIS:\033[0m")
    rss_spline = np.sum((y_smooth - np.mean(y_smooth)) ** 2)
    n_permutations = 10000
    rss_random = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_permutations):
            y_shuffled = np.random.permutation(y)
            try:
                spline_shuffled = UnivariateSpline(x, y_shuffled, k=3, s=len(x) * 2.0)
                y_shuffled_smooth = spline_shuffled(x_smooth)
                rss = np.sum((y_shuffled_smooth - np.mean(y_shuffled_smooth)) ** 2)
                rss_random.append(rss)
            except:
                continue
    p_value = np.mean(np.array(rss_random) > rss_spline)
    print(f"Permutation p-value: {p_value:.8f}")
    print(f"R² = {r_squared:.4f}")

    # Plot spline + raw data
    x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_series['date'], y, marker='o', linestyle='-', color=line_color_data, alpha=0.7, label='Observed Data')
    plt.plot(x_smooth_dates, y_smooth, linestyle='--', linewidth=2.5, color='teal', alpha=0.9, label='Cubic Spline Fit')
    plt.text(r_p_pos[0], r_p_pos[1], f"R² = {r_squared:.2f}", color='teal', fontsize=21, fontweight='bold', ha='right', va='top', transform=plt.gca().transAxes)
    plt.text(r_p_pos[0], r_p_pos[1] - 0.08, f"Permutation test\np = {p_value:.3g}", color='teal', fontsize=18, ha='right', va='top', transform=plt.gca().transAxes)
    plt.title(f"Cubic Spline Model vs Raw Observations: {theme} ({anx_or_hap})", loc='left', fontsize=16)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    cubic_save_path = 'E:/twitter_analysis_data/viz/theme_analysis/cubic_spline_fitting/'
    plt.savefig(cubic_save_path + "cubic fit " + anx_or_hap.lower() + " " + theme + '.png', bbox_inches='tight', pad_inches=0.15)
    plt.show()
    plt.close("all")
    
    # Change point detection
    print("\n\033[1mCHANGE POINT DETECTION:\033[0m")
    algo = rpt.Pelt(model="rbf").fit(y_smooth)
    change_points = algo.predict(pen=20)
    change_dates = x_smooth_dates[change_points[:-1]]
    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth_dates, y_smooth, label="Cubic spline model", linewidth=2, color=line_color_data)
    
    if len(change_dates) > 0:
        for cp in change_dates:
            plt.axvline(cp, color='red', linestyle='--', alpha=0.7)
        # Only label the first one
        plt.axvline(change_dates[0], color='red', linestyle='--', alpha=0.7, label='Change Point')
    else:
        print("No change points detected.")
    
    plt.title(f"Change Point Detection: {theme} ({anx_or_hap})", loc='left', fontsize=16)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    change_point_save_path = 'E:/twitter_analysis_data/viz/theme_analysis/change_point_identification/'
    plt.savefig(change_point_save_path + "change point " + anx_or_hap.lower() + " " + theme + '.png', bbox_inches='tight', pad_inches=0.15)
    plt.show()

    print("\n \033[1mDEFINED PERIODS\033[0m")
    for p in periods:
        print(p)
    
    # logistic regression and GEE modelling of periods, with T-test and effect size
    print("\n\n \033[1mLOGISTIC REGRESSION, T-TESTING AND GEE MODELLING\033[0m \n")
    input_dataframe = input_dataframe.copy()
    input_dataframe["month"] = pd.to_datetime(input_dataframe["date"]).dt.to_period("M").dt.to_timestamp()
    input_dataframe["theme_binary"] = input_dataframe[theme].map({"Y": 1, "N": 0})
    input_dataframe["period"] = pd.Series([None] * len(input_dataframe), dtype="object")

    for label, start, end in periods:
        start_date = pd.to_datetime(start + "-01")
        end_date = pd.to_datetime(end + "-01")
        input_dataframe.loc[
            (input_dataframe["date"] >= start_date) & (input_dataframe["date"] <= end_date),
            "period"
        ] = label

    input_dataframe = input_dataframe.dropna(subset=["period"])
    input_dataframe["month_index"] = input_dataframe.groupby("period")["month"].transform(lambda x: (x - x.min()).dt.days // 30)
    raw_monthly = input_dataframe.groupby("month")["theme_binary"].mean().reset_index(name="observed_rate")
    plt.figure(figsize=(12, 6))
    plt.plot(raw_monthly["month"], raw_monthly["observed_rate"], marker='o', linestyle='-', color=line_color_data, alpha=0.5)
    model_summaries = []
    binary_data_by_period = {}

    for period in input_dataframe["period"].unique():
        period_df = input_dataframe[input_dataframe["period"] == period].copy()

        model = logit("theme_binary ~ month_index", data=period_df).fit(disp=False)
        intercept = model.params["Intercept"]
        slope = model.params["month_index"]
        p_intercept = model.pvalues["Intercept"]
        p_slope = model.pvalues["month_index"]
        period_df["predicted_prob"] = model.predict(period_df)

        x_vals = np.arange(0, period_df["month_index"].max() + 1)
        x_dates = pd.date_range(start=period_df["month"].min(), periods=len(x_vals), freq="MS")
        log_odds = intercept + slope * x_vals
        probabilities = 1 / (1 + np.exp(-log_odds))
        plt.plot(x_dates, probabilities, linestyle='--', linewidth=2, color='teal')

        if period in logit_text_positions:
            label_x = x_dates[int(logit_text_positions[period][0] * len(x_dates))]
            label_y = logit_text_positions[period][1]
        else:
            middle_idx = len(x_dates) // 2
            label_x = x_dates[middle_idx]
            label_y = probabilities[middle_idx] + 0.03

        expression = f"{period}\n" + r"$p = \frac{1}{1 + e^{-(" + f"{intercept:.2f} + {slope:.2f} · t" + r")}}$"
        plt.text(label_x, label_y, expression, color='teal', fontsize=14, ha='center', va='bottom')

        empirical_mean = period_df["theme_binary"].mean()

        model_summaries.append({
            "Period": period,
            "Intercept": intercept,
            "Slope": slope,
            "p(Intercept)": p_intercept,
            "p(Slope)": p_slope,
            "Empirical Mean": empirical_mean,
        })

        binary_data_by_period[period] = period_df["theme_binary"]

    plt.title(f"Logistic Regression for Periods: {theme} ({anx_or_hap})", loc='left', fontsize=16, color="black", fontweight='normal')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    log_reg_save_path = 'E:/twitter_analysis_data/viz/theme_analysis/logistic_reg_by_period/'
    plt.savefig(log_reg_save_path + "log reg " + anx_or_hap.lower() + " " + theme + '.png', bbox_inches='tight', pad_inches=0.15)
    plt.show()

    model_summary_df = pd.DataFrame(model_summaries)
    print("\n\n \033[1mSummary of Logistic Regression Model and Means for Periods:\033[0m")
    print(model_summary_df)

    print("\n\nT-TESTING TO ASSESS VARIATION IN MEANS BETWEEN PERIODS\n")
    periods_list = list(binary_data_by_period.keys())
    pvals = pd.DataFrame(index=periods_list, columns=periods_list, dtype=float)

    for i, p1 in enumerate(periods_list):
        for j, p2 in enumerate(periods_list):
            if i == j:
                pvals.loc[p1, p2] = np.nan
            else:
                stat, pval = stats.ttest_ind(binary_data_by_period[p1], binary_data_by_period[p2], equal_var=False)
                pvals.loc[p1, p2] = pval

    #plt.figure(figsize=(6, 5))
    #cmap = sns.light_palette("teal", as_cmap=True)
    #sns.heatmap(pvals.astype(float), annot=True, fmt=".8f", cmap=cmap, cbar_kws={'label': 'p-value'})
    #plt.title(f"T-Testing: {theme} ({anx_or_hap})", loc='left', fontsize=14)
    #plt.tight_layout()
    #plt.show()

    print("\n\033[1mPairwise T-Test p-values using Welch's T-Test:\033[0m")
    print(pvals)

    print("\n \033[1m Calculation of effect size using Cohen's d:\033[0m")
    effect_sizes = []
    for i, p1 in enumerate(periods_list):
        for j, p2 in enumerate(periods_list):
            if i < j:
                mean1, mean2 = binary_data_by_period[p1].mean(), binary_data_by_period[p2].mean()
                std1, std2 = binary_data_by_period[p1].std(), binary_data_by_period[p2].std()
                n1, n2 = len(binary_data_by_period[p1]), len(binary_data_by_period[p2])
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan
                effect_sizes.append({"Comparison": f"{p1} vs {p2}", "Cohen's d": d})

    effect_sizes_df = pd.DataFrame(effect_sizes)
    print(effect_sizes_df)

    print("\n\nGEE MODELLING FOR EACH PERIOD\n")
    gee_results = []
    plt.figure(figsize=(12, 6))

    plt.plot(raw_monthly["month"], raw_monthly["observed_rate"], marker='o', linestyle='-', color=line_color_data, alpha=0.5)

    for period in periods_list:
        period_df = input_dataframe[input_dataframe["period"] == period].copy()

        model_gee = GEE.from_formula(
            "theme_binary ~ month_index",
            groups=period_df["user_id"],
            data=period_df,
            family=Binomial(),
            cov_struct=Exchangeable()
        ).fit()

        coef = model_gee.params['month_index']
        intercept = model_gee.params["Intercept"]
        pval = model_gee.pvalues['month_index']

        gee_results.append({"Period": period, "GEE Slope": coef, "GEE p-value": pval})

        prediction_months = pd.date_range(
            start=period_df["month"].min(), end=period_df["month"].max(), freq='MS'
        )
        month_index_pred = (prediction_months - period_df["month"].min()).days // 30

        log_odds_pred = intercept + coef * month_index_pred
        prob_pred = 1 / (1 + np.exp(-log_odds_pred))

        plt.plot(
            prediction_months,
            prob_pred,
            linestyle='--',
            linewidth=2,
            color='teal'
        )

    for period in periods_list:
        period_df = input_dataframe[input_dataframe["period"] == period]
        if not period_df.empty:
            midpoint_month = period_df["month"].min() + (period_df["month"].max() - period_df["month"].min()) / 2
            midpoint_value = period_df["theme_binary"].mean() + gee_text_positions.get(period, 0.05)
            #plt.text(midpoint_month, midpoint_value, period, color="teal", fontsize=14, ha='center', va='bottom')

    plt.title(f"GEE Model Fits for Periods: {theme} ({anx_or_hap})", loc='left', fontsize=16, color="black", fontweight='normal')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    gee_summary_df = pd.DataFrame(gee_results)
    print("\n \033[1m Generalized Estimating Equations (GEE) Analysis:\033[0m")
    print(gee_summary_df)

    #return model_summary_df, pvals, effect_sizes_df, gee_summary_df


# ## Major Drivers of Anxiety
# ---

# ### Covid Pandemic (Anxiety and Happiness)
# ---

# In[ ]:


### CREATE WC / LINE VIZ : ANX - COIVD PANDEMIC
## covid tiemline: https://www.instituteforgovernment.org.uk/sites/default/files/2022-12/timeline-coronavirus-lockdown-december-2021.pdf

# ----------> SELECT THEME --------
theme = 'Covid Pandemic'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

### ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create an interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 2.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth_viz[y_smooth_viz < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()


## ----- Display y-axis labels 
max_rate = 50
min_rate = 0
mid_rate = 25
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 1, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 1, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 1, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.21, 0.825,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 1 (retail price index peak) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_mar_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_covid_lockdown_png.png')
# define the position and size parameters
ld_mar_image_xaxis = 0.225
ld_mar_image_yaxis = 0.68
ld_mar_image_width = 0.07
ld_mar_image_height = 0.07
ax_ld_mar_image = fig.add_axes([ld_mar_image_xaxis, ld_mar_image_yaxis, ld_mar_image_width, ld_mar_image_height])
# show icon
ax_ld_mar_image.imshow(ld_mar_image)
ax_ld_mar_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.175, 1.04,  # X, Y in Axes coordinates
    r"$\bf{Mar\ 2020:}$" + "\nFirst national\nlockdown",
    #f"Oct 2022:\nRetail Price Index\npeaks at 11.1%.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.36, 0.734), (0.467, 0.69))

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_covid_lockdown_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.605
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.07
ld_nov_image_height = 0.07
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.72, 0.83,  # X, Y in Axes coordinates
    r"$\bf{Nov'20 - Jan'21:}$" + "\nSecond and Third\nNational Lockdowns",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=False):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.62, 0.62), (0.555, 0.54))

##### ICON, TEXT, ARROW 3 (plan b measures) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
planb_dec_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_covid_planb_png.png')
# define the position and size parameters
planb_dec_image_xaxis = 0.805
planb_dec_image_yaxis = 0.33
planb_dec_image_width = 0.08
planb_dec_image_height = 0.08
ax_planb_dec_image = fig.add_axes([planb_dec_image_xaxis, planb_dec_image_yaxis, planb_dec_image_width, planb_dec_image_height])
# show icon
ax_planb_dec_image.imshow(planb_dec_image)
ax_planb_dec_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.94, 0.19,  # X, Y in Axes coordinates
    r"$\bf{Dec\ 2021:}$" + "\nPlan B Measures\nImplemented",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=False):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.75, 0.4), (0.678, 0.435))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

###################################################### ANALYSIS AND TESTING ######################################

stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme="Covid Pandemic",
    periods=[
        ("Pre-pandemic", "1900-01", "2020-01"),
        ("Pandemic", "2020-02", "2021-03"),
        ("Post-pandemic", "2021-04", "2100-01"),
    ],
    logit_text_positions={
        "Pre-pandemic": (0.3, 0.07),
        "Pandemic": (0.655, 0.35),
        "Post-pandemic": (0.75, 0.07)
    },
    gee_text_positions={
        "Pre-pandemic": 0.05,
        "Pandemic": 0.05,
        "Post-pandemic": 0.05
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.95)
)



# In[ ]:


### CREATE WC / LINE VIZ : HAP - COIVD PANDEMIC
## covid tiemline: https://www.instituteforgovernment.org.uk/sites/default/files/2022-12/timeline-coronavirus-lockdown-december-2021.pdf

# ----------> SELECT THEME --------
theme = 'Covid Pandemic'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 0.01)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)


# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 2
min_rate = 0
mid_rate = 1
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.04, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.04, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.04, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.22, 0.825,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_cov_open_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.45
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.415, 0.83,  # X, Y in Axes coordinates
    r"$\bf{March\ 2021:}$" + "\nSchools reopen and\ngatherings allowed.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.52, 0.62), (0.58, 0.57))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-pandemic"
p2 = "Pandemic"
p3 = "Post-pandemic"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "1900-01", "2020-01"),
        (p2, "2020-02", "2021-03"),
        (p3, "2021-04", "2100-01")
    ],
    logit_text_positions={
        p1: (0.5, 0.005),
        p2: (0.5, 0.004),
        p3: (0.655, 0.01)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# ### Personal Finances and Employment (Anxiety)
# ---

# In[ ]:


### CREATE WC / LINE VIZ : ANX - PERSONAL FINANCE AND EMPLOYMENT

# ----------> SELECT THEME --------
theme = 'Personal Finances and Employment'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 2.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 25
min_rate = 10
mid_rate_1 = 15
mid_rate_2 = 20

sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_1, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_2, color='black', linestyle='--', lw=0.4)

sub_ax.text(17350, max_rate - 0.22, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.22, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_1 - 0.22, f"{mid_rate_1:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_2 - 0.22, f"{mid_rate_2:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")
#f"""
#Are movies getting <worse>?
#<Average rating per year on {df.shape[0]} movies published between {lower_bound} and {upper_bound}>
#"""

fig_text(
    0.32, 0.825,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 1 (retail price index peak) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
rpi_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_pfe_rpi_png.png')
# define the position and size parameters
rpi_image_xaxis = 0.5
rpi_image_yaxis = 0.68
rpi_image_width = 0.08
rpi_image_height = 0.08
ax_rpi_image = fig.add_axes([rpi_image_xaxis, rpi_image_yaxis, rpi_image_width, rpi_image_height])
# show icon
ax_rpi_image.imshow(rpi_image)
ax_rpi_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.577, 1.063,  # X, Y in Axes coordinates
    r"$\bf{Oct\ 2022:}$" + "\nRetail Price Index\npeaks at 11.1%",
    #f"Oct 2022:\nRetail Price Index\npeaks at 11.1%.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.635, 0.745), (0.78, 0.675))

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
col_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_pfe_col_png.png')
# define the position and size parameters
col_image_xaxis = 0.82
col_image_yaxis = 0.4
col_image_width = 0.08
col_image_height = 0.08
ax_col_image = fig.add_axes([col_image_xaxis, col_image_yaxis, col_image_width, col_image_height])
# show icon
ax_col_image.imshow(col_image)
ax_col_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.96, 0.352,  # X, Y in Axes coordinates
    r"$\bf{Nov\ 2022:}$" + "\nCost of Living\npayments",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)

draw_arrow((0.83, 0.47), (0.796, 0.588))

sub_ax.text(
    0.3, 0.1,  # X, Y in Axes coordinates
    r"$\bf{Oct\ 2019}$" + "\n\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

def draw_arrow(tail_position, head_position, invert=False):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.375, 0.353), (0.405, 0.42))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Stability to Feb'20" 
p2 = "Gradual Incline Feb'20-Oct'22"
p3 = "Decline from Oct'22"


stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-02", "2022-10"),
        (p3, "2022-10", "2023-03"),
    ],
    logit_text_positions={
        p1: (0.3, 0.12),
        p2: (0.5, 0.23),
        p3: (0.3, 0.15)
    },
    gee_text_positions={
        p1: 0.05,
        p2: 0.05,
        p3: 0.05
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.21)
)



# ### Business (Anxiety and Happiness)

# In[ ]:


### CREATE WC / LINE VIZ : ANX - BUSINESS

# ----------> SELECT THEME --------
theme = 'Business'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 0.5)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)


# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth_viz[y_smooth_viz < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 10
min_rate = 0
mid_rate = 5
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.2, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.2, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.2, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')


# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.18, 0.819,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')



##### ICON, TEXT, ARROW 1 (invasion of ukraine) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
business_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_business_png.png')
# define the position and size parameters
business_image_xaxis = 0.425
business_image_yaxis = 0.357
business_image_width = 0.08
business_image_height = 0.08
ax_business_image = fig.add_axes([business_image_xaxis, business_image_yaxis, business_image_width, business_image_height])
# show icon
ax_business_image.imshow(business_image)
ax_business_image.axis('off')  # Remove axis of the image

### 4 PEAKS: Mar20, Jul20, Nov20, Apr21

## ----- Icon text
sub_ax.text(
    0.47, 0.265,  # X, Y in Axes coordinates
    r"$\bf{Mar'20-Apr'21:}$" + "\nBusiness closure\nthrough lockdowns.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ---- Texy: Mar 2020
sub_ax.text(
    0.39, 0.78,  # X, Y in Axes coordinates
    r"$\bf{Mar'20}$" + "\n\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ---- Texy: Jul 2020
sub_ax.text(
    0.455, 0.95,  # X, Y in Axes coordinates
    r"$\bf{Jul'20}$" + "\n\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ---- Texy: Nov 2020
sub_ax.text(
    0.525, 0.93,  # X, Y in Axes coordinates
    r"$\bf{Nov'20}$" + "\n\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ---- Texy: Mar 2021
sub_ax.text(
    0.59, 0.715,  # X, Y in Axes coordinates
    r"$\bf{Apr'21}$" + "\n\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.62, 0.61), (0.702, 0.582))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre Lockdowns"
p2 = "Business Closures"
p3 = "Post Lockdowns"

stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-01"),
        (p2, "2020-01", "2021-05"),
        (p3, "2021-05", "2023-03"),
    ],
    logit_text_positions={
        p1: (0.3, 0.07),
        p2: (0.655, 0.1),
        p3: (0.75, 0.07)
    },
    gee_text_positions={
        p1: 0.05,
        p2: 0.05,
        p3: 0.05
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.95)
)



# In[ ]:


### CREATE WC / LINE VIZ : HAP - BUSINESS

# ----------> SELECT THEME --------
theme = 'Business'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 0.5)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 10
min_rate = 4
mid_rate_1 = 6
mid_rate_2 = 8
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_1, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_2, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.04, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.04, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_1 - 0.04, f"{mid_rate_1:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_2 - 0.04, f"{mid_rate_2:.0f}%",
    fontsize=10, color='black', fontweight='bold')


# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.18, 0.815,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_cov_open_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.45
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
#ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

# peakes: Jul 2020, Nov 2020, April 2021

## --Text Jul20
sub_ax.text(
    0.51, 0.94,  # X, Y in Axes coordinates
    r"$\bf{Jul'20}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## --Text Nov20
sub_ax.text(
    0.58, 0.88,  # X, Y in Axes coordinates
    r"$\bf{Nov'20}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## --Text Apr21
sub_ax.text(
    0.66, 0.851,  # X, Y in Axes coordinates
    r"$\bf{Apr'21}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.52, 0.62), (0.58, 0.57))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre Lockdowns"
p2 = "Business Closures"
p3 = "Post Lockdowns"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-01"),
        (p2, "2020-01", "2021-05"),
        (p3, "2021-05", "2023-03"),
    ],
    logit_text_positions={
        p1: (0.5, 0.025),
        p2: (0.5, 0.07),
        p3: (0.655, 0.08)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# ### Work Life Anxiety and Happiness)

# In[ ]:


### CREATE WC / LINE VIZ : ANX - WORK LIFE
## working from home stats: https://thehomeofficelife.com/blog/work-from-home-statistics
# ons: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/articles/characteristicsofhomeworkersgreatbritain/september2022tojanuary2023
# ----------> SELECT THEME --------
theme = 'Work Life'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth_viz[y_smooth_viz < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 15
min_rate = 0
mid_rate_1 = 5
mid_rate_2 = 10

sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_1, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_2, color='black', linestyle='--', lw=0.4)

sub_ax.text(17350, max_rate - 0.25, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.25, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_1 - 0.25, f"{mid_rate_1:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_2 - 0.25, f"{mid_rate_2:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.185, 0.821,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 1 (furlough and wfh peak) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
fur_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_work_fur_png.png')
# define the position and size parameters
fur_image_xaxis = 0.37
fur_image_yaxis = 0.35
fur_image_width = 0.075
fur_image_height = 0.075
ax_fur_image = fig.add_axes([fur_image_xaxis, fur_image_yaxis, fur_image_width, fur_image_height])
# show icon
ax_fur_image.imshow(fur_image)
ax_fur_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.31, 0.225,  # X, Y in Axes coordinates
    r"$\bf{Apr'20-Jun'20:}$" + "\nFurlough peaks at 8.9 million\nWFH peaks at 39%-49%",
    #f"Oct 2022:\nRetail Price Index\npeaks at 11.1%.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=False):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.43, 0.38), (0.471, 0.429))

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
hybrid_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_work_hybrid_png.png')
# define the position and size parameters
hybrid_image_xaxis = 0.59
hybrid_image_yaxis = 0.403
hybrid_image_width = 0.075
hybrid_image_height = 0.075
ax_hybrid_image = fig.add_axes([hybrid_image_xaxis, hybrid_image_yaxis, hybrid_image_width, hybrid_image_height])
# show icon
ax_hybrid_image.imshow(hybrid_image)
ax_hybrid_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.715, 0.359,  # X, Y in Axes coordinates
    r"$\bf{Post-Pandemic\ Period:}$" + "\nHybrid working culture\nacross many industries.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=False):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.62, 0.62), (0.555, 0.5325))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-Furlough"
p2 = "Furlough"
p3 = "Hybrid Working"

stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-03", "2021-01"),
        (p3, "2021-02", "2023-03"),
    ],
    logit_text_positions={
        p1: (0.3, 0.085),
        p2: (0.655, 0.1),
        p3: (0.7, 0.06)
    },
    gee_text_positions={
        p1: 0.05,
        p2: 0.05,
        p3: 0.05
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.21)
)



# In[ ]:


### CREATE WC / LINE VIZ : HAP - WORK LIFE

# ----------> SELECT THEME --------
theme = 'Work Life'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 6
min_rate = 2
mid_rate = 4
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)

sub_ax.text(17350, max_rate - 0.125, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.125, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.125, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.18, 0.815,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### TEXT 1 (June 20) -------------------------------------------------------------------------------------------

## ----- Icon text
sub_ax.text(
    0.31, 0.95,  # X, Y in Axes coordinates
    r"$\bf{Oct\ 2019}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

sub_ax.text(
    0.425, -0.05,  # X, Y in Axes coordinates
    r"$\bf{Jun\ 2020}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


# ### Public Services (Anxiety and Happiness)

# In[ ]:


### CREATE WC / LINE VIZ : ANX - PUBLIC SERVICES
## war in ukraine timeline: https://researchbriefings.files.parliament.uk/documents/CBP-9847/CBP-9847.pdf

# ----------> SELECT THEME --------
theme = 'Public Services'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function

x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 15
min_rate = 5
mid_rate = 10
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.25, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.25, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.25, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.21, 0.821,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')



##### ICON, TEXT, ARROW 1 (invasion of ukraine) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
pub_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_pub_png.png')
# define the position and size parameters
pub_image_xaxis = 0.445
pub_image_yaxis = 0.585
pub_image_width = 0.075
pub_image_height = 0.075
ax_pub_image = fig.add_axes([pub_image_xaxis, pub_image_yaxis, pub_image_width, pub_image_height])
# show icon
ax_pub_image.imshow(pub_image)
ax_pub_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.5, 0.826,  # X, Y in Axes coordinates
    r"$\bf{Mar'20\ Onwards:}$" + "\nIncreased pressure\non public services.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.62, 0.61), (0.702, 0.582))


# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-2020"
p2 = "2020 Onwards"

stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-03", "2023-03"),
    ],
    logit_text_positions={
        p1: (0.7, 0.09),
        p2: (0.4, 0.07),
    },
    gee_text_positions={
        p1: 0.05,
        p2: 0.05,
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.21)
)



# In[ ]:


### CREATE WC / LINE VIZ : HAP - PUBLIC SERVICES

# ----------> SELECT THEME --------
theme = 'Public Services'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 0.75)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 15
min_rate = 5
mid_rate = 10
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.2, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.2, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.2, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')


# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.219, 0.817,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_pub_clap_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.58
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.62, 0.83,  # X, Y in Axes coordinates
    r"$\bf{Mar'20-May'20}$" + "\nClap for Heroes\nmovement.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.522, 0.612), (0.475, 0.6))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-pandemic"
p2 = "Pandemic"
p3 = "Post-Pandemic"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-03", "2021-12"),
        (p3, "2022-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.05),
        p2: (0.5, 0.05),
        p3: (0.655, 0.05)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# ### Safety and Crime (Anxiety)

# In[ ]:


### CREATE WC / LINE VIZ : ANX - SAFTEY AND CRIME

# ----------> SELECT THEME --------
theme = 'Safety and Crime'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud background
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)* 1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 20
min_rate = 5
mid_rate_1 = 10
mid_rate_2 = 15
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_1, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_2, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.2, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.2, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_1 - 0.2, f"{mid_rate_1:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_2 - 0.2, f"{mid_rate_2:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.225, 0.819,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 1 (invasion of ukraine) -------------------------------------------------------------------------------------------

## ----- Icon text
sub_ax.text(
    0.385, 0.9,  # X, Y in Axes coordinates
    r"$\bf{Mar\ 2020}$" + "\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.62, 0.61), (0.702, 0.582))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "2018-2019"
p2 = "2020"
p3 = "2021 Onwards"


stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2019-12"),
        (p2, "2020-01", "2020-12"),
        (p3, "2021-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.1),
        p2: (0.5,0.07),
        p3: (0.655, 0.15)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.95)
)


# ### Geopolitics (Anxiety)

# In[ ]:


### CREATE WC / LINE VIZ : ANX - GEOPOLITICS
## war in ukraine timeline: https://researchbriefings.files.parliament.uk/documents/CBP-9847/CBP-9847.pdf

# ----------> SELECT THEME --------
theme = 'Geopolitics'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*0.25)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 15
min_rate = 0
mid_rate_1 = 5
mid_rate_2 = 10

sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_1, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_2, color='black', linestyle='--', lw=0.4)

sub_ax.text(17350, max_rate - 0.25, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.25, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_1 - 0.25, f"{mid_rate_1:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_2 - 0.25, f"{mid_rate_2:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.185, 0.821,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')


##### ICON, TEXT, ARROW 1 (invasion of ukraine) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
war_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/anx_geopol_war_png.png')
# define the position and size parameters
war_image_xaxis = 0.445
war_image_yaxis = 0.54
war_image_width = 0.075
war_image_height = 0.075
ax_war_image = fig.add_axes([war_image_xaxis, war_image_yaxis, war_image_width, war_image_height])
# show icon
ax_war_image.imshow(war_image)
ax_war_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.5, 0.71,  # X, Y in Axes coordinates
    r"$\bf{Feb'22-Mar'22:}$" + "\nRussian invasion\nof Ukraine.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.62, 0.61), (0.702, 0.582))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre UA Invasion"
p2 = "UA Invasion"
p3 = "POST UA Invasion"


stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2022-01"),
        (p2, "2022-01", "2022-04"),
        (p3, "2022-04", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.025),
        p2: (0.5,0.07),
        p3: (0.655, 0.025)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data="grey",
    r_p_pos = (0.98, 0.95)
)


# ## Major Drivers of Happiness
# ---
# 

# ### Celeration and Seasonal (Happiness)

# In[ ]:


### CREATE WC / LINE VIZ : HAP  - CELEBRATION AND SEASONAL

# ----------> SELECT THEME --------
theme = 'Celebration and Seasonal'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*2.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)


# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 50
min_rate = 0
#mid_rate = 8
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
#sub_ax.axhline(y=mid_rate, color='lightgrey', linestyle='--', lw=0.6)
sub_ax.text(17350, max_rate - 0.8, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.8, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.27, 0.815,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output


##### ICON 1 (tree) -------------------------------------------------------------------------------------------
# initialise image object from file
tree_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_celeb_tree_png.png')
# define the position and size parameters
tree_image_xaxis = 0.268
tree_image_yaxis = 0.658
tree_image_width = 0.08
tree_image_height = 0.08
ax_tree_image = fig.add_axes([tree_image_xaxis, tree_image_yaxis, tree_image_width, tree_image_height])
# show icon
ax_tree_image.imshow(tree_image)
ax_tree_image.axis('off')  # Remove axis of the image

##### ICON 2 (gift) -------------------------------------------------------------------------------------------

# initialise image object from file
gift_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_celeb_gift_png.png')
# define the position and size parameters
gift_image_xaxis = 0.3965
gift_image_yaxis = 0.635
gift_image_width = 0.07
gift_image_height = 0.07
ax_gift_image = fig.add_axes([gift_image_xaxis, gift_image_yaxis, gift_image_width, gift_image_height])
# show icon
ax_gift_image.imshow(gift_image)
ax_gift_image.axis('off')  # Remove axis of the image

##### ICON 3 (candy) -------------------------------------------------------------------------------------------

# initialise image object from file
candy_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_celeb_candy_png.png')
# define the position and size parameters
candy_image_xaxis = 0.5195
candy_image_yaxis = 0.5952
candy_image_width = 0.07
candy_image_height = 0.07
ax_candy_image = fig.add_axes([candy_image_xaxis, candy_image_yaxis, candy_image_width, candy_image_height])
# show icon
ax_candy_image.imshow(candy_image)
ax_candy_image.axis('off')  # Remove axis of the image

##### ICON 4 (man) -------------------------------------------------------------------------------------------

# initialise image object from file
man_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_celeb_man_png.png')
# define the position and size parameters
man_image_xaxis = 0.637
man_image_yaxis = 0.625
man_image_width = 0.08
man_image_height = 0.08
ax_man_image = fig.add_axes([man_image_xaxis, man_image_yaxis, man_image_width, man_image_height])
# show icon
ax_man_image.imshow(man_image)
ax_man_image.axis('off')  # Remove axis of the image

##### ICON 5 (holly) -------------------------------------------------------------------------------------------

# initialise image object from file
holly_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_celeb_holly_png.png')
# define the position and size parameters
holly_image_xaxis = 0.759
holly_image_yaxis = 0.6125
holly_image_width = 0.08
holly_image_height = 0.08
ax_holly_image = fig.add_axes([holly_image_xaxis, holly_image_yaxis, holly_image_width, holly_image_height])
# show icon
ax_holly_image.imshow(holly_image)
ax_holly_image.axis('off')  # Remove axis of the image


# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "n/a"
p2 = "n/a"
p3 = "n/a"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-01"),
        (p2, "2020-01", "2021-12"),
        (p3, "2022-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.025),
        p2: (0.5,0.07),
        p3: (0.655, 0.025)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# ### Social and Community (Happiness and Anxiety)

# In[ ]:


### CREATE WC / LINE VIZ : HAP - PERSONAL

# ----------> SELECT THEME --------
theme = 'Social and Community'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 80
min_rate = 70
mid_rate = 75
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.2, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.2, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.2, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.25, 0.817,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_cov_open_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.45
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
#ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.475, 0.95,  # X, Y in Axes coordinates
    r"$\bf{Mar\ 2020}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

sub_ax.text(
    0.858, 0.745,  # X, Y in Axes coordinates
    r"$\bf{May\ 2022}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.52, 0.62), (0.58, 0.57))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "2018-2019"
p2 = "2020-2021"
p3 = "2022 and up to Apr 2023"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-01"),
        (p2, "2020-01", "2021-12"),
        (p3, "2022-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.75),
        p2: (0.5, 0.75),
        p3: (0.655, 0.75)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# In[ ]:


### CREATE WC / LINE VIZ : ANX - SOCIAL AND COMMUNITY

# ----------> SELECT THEME --------
theme = 'Social and Community'
# ---------------------------------

anx_wc_count_dict = anx_longitudinal_df[anx_longitudinal_df['theme'] == theme]['anx_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
def adjusted_grey_scale_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    max_freq = max(anx_wc_count_dict.values())  # Get max frequency for scaling
    frequency = anx_wc_count_dict.get(word, 1)
    min_darkness = 230  # Adjust this to make the lightest words darker
    darkness = int(200 * (1 - frequency / max_freq))  # Scale from black (0) to white (255)
    darkness = max(darkness, min_darkness)  # Ensure minimum darkness level
    return f"rgb({darkness}, {darkness}, {darkness})"  # Return adjusted grayscale color

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=2,
    color_func=adjusted_grey_scale_color_func
).generate_from_frequencies(anx_wc_count_dict)
# show wordcloud background
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = anx_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='darkgrey', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 35
min_rate = 15
mid_rate_1 = 25
#mid_rate_2 = 25
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate_1, color='black', linestyle='--', lw=0.4)
#sub_ax.axhline(y=mid_rate_2, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.2, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.2, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate_1 - 0.2, f"{mid_rate_1:.0f}%",
    fontsize=10, color='black', fontweight='bold')
#sub_ax.text(17350, mid_rate_2 - 0.2, f"{mid_rate_2:.0f}%",
#    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.25, 0.819,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 1 (invasion of ukraine) -------------------------------------------------------------------------------------------

## ----- Icon text
sub_ax.text(
    0.43, 0.885,  # X, Y in Axes coordinates
    r"$\bf{Jun\ 2020}$" + "\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

sub_ax.text(
    0.39, 0.05,  # X, Y in Axes coordinates
    r"$\bf{Mar\ 2020}$" + "\n",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')


## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.62, 0.61), (0.702, 0.582))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "anx " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "2018-2019"
p2 = "2020-2021"
p3 = "2022 and up to Apr 2023"

stat_analysis_for_theme(
    anx_or_hap = "Anxiety",
    input_dataframe=anx_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-01"),
        (p2, "2020-01", "2021-12"),
        (p3, "2022-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.4),
        p2: (0.5, 0.4),
        p3: (0.655, 0.4)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='grey',
    r_p_pos = (0.98, 0.95)
)


# ### Recreation (Events, Food and Drink, Travel and Transport) (Happiness)

# In[ ]:


### CREATE WC / LINE VIZ : HAP - EVENTS
## covid tiemline: https://www.instituteforgovernment.org.uk/sites/default/files/2022-12/timeline-coronavirus-lockdown-december-2021.pdf

# ----------> SELECT THEME --------
theme = 'Events'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 10
min_rate = 0
mid_rate = 5
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.15, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.15, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.15, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')

# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.17, 0.824,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_cov_open_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.45
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
#ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.4, 0.05,  # X, Y in Axes coordinates
    r"$\bf{April\ 2020}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.52, 0.62), (0.58, 0.57))


# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-Pandemic"
p2 = "Pandemic"
p3 = "Post-Pendemic"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-03", "2021-05"),
        (p3, "2021-06", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.025),
        p2: (0.5, 0.07),
        p3: (0.655, 0.08)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# In[ ]:


### CREATE WC / LINE VIZ: HAP - FOOD AND DRINK

# ----------> SELECT THEME --------
theme = 'Food and Drink'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 15
min_rate = 5
mid_rate = 10
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.125, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.125, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.125, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')


# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.22, 0.824,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### TEXT 1 (June 20) -------------------------------------------------------------------------------------------

## ----- Icon text
sub_ax.text(
    0.42, 0.01,  # X, Y in Axes coordinates
    r"$\bf{June\ 2020}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

##### ICON, TEXT, ARROW 2 (eat out to help out) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_food_eat_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.565
ld_nov_image_yaxis = 0.64
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.69, 0.95,  # X, Y in Axes coordinates
    r"$\bf{Aug\ 2020:}$" + "\nEat out to help\nout scheme.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=False):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
draw_arrow((0.57, 0.67), (0.515, 0.58))


# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################


p1 = "Pre-Pandemic"
p2 = "Pandemic"
p3 = "Post-Pendemic"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-03", "2020-12"),
        (p3, "2021-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.07),
        p2: (0.5, 0.1),
        p3: (0.655, 0.09)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# In[ ]:


### CREATE WC / LINE VIZ : HAP - TRAVEL AND TRANSPORT

# ----------> SELECT THEME --------
theme = 'Travel and Transport'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*0.25)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 6
min_rate = 0
mid_rate = 3
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.125, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.125, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.125, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')


# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.236, 0.824,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### TEXT 1 (June 20) -------------------------------------------------------------------------------------------

## ----- Icon text
sub_ax.text(
    0.413, 0.047,  # X, Y in Axes coordinates
    r"$\bf{May\ 2020}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='left',
    va='center')

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()


################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-Pandemic"
p2 = "Pandemic"
p3 = "Post-Pendemic"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-02"),
        (p2, "2020-03", "2020-12"),
        (p3, "2021-01", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.02),
        p2: (0.5, 0.04),
        p3: (0.655, 0.01)
    },
    gee_text_positions={
        p1: 0.05,
        p3: 0.05,
        p3: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# ### Personal (Happiness)

# In[ ]:


### CREATE WC / LINE VIZ : HAP - PERSONAL

# ----------> SELECT THEME --------
theme = 'Personal'
# ---------------------------------

hap_longitudinal_df.loc[:,'hap_topics_cleansed'] = [s.replace('celebrations','celebration') for s in hap_longitudinal_df['hap_topics_cleansed']]

hap_wc_count_dict = hap_longitudinal_df[hap_longitudinal_df['theme'] == theme]['hap_topics_cleansed'].value_counts().to_dict()

## ------ Initiate a figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

## ------ Define a function to control colour darkness for background wc
# Define a light version of #FFC000 — you can tweak this RGB as desired
light_yellow_rgb = (255, 245, 180)  # lighter tint of #FFC000

def constant_light_yellow_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    r, g, b = light_yellow_rgb
    return f"rgb({r}, {g}, {b})"

## ----- Generate the word cloud with adjusted colours
# initialise wc object
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color='white',
    contour_width=1,
    max_font_size=300,
    relative_scaling=0.2,
    margin=1,
    collocations=False,
    random_state=11,               
    color_func=constant_light_yellow_color_func
).generate_from_frequencies(hap_wc_count_dict)
# show wordcloud backgorund
ax.imshow(wordcloud)
ax.set_axis_off()

## ----- Create time series of % measure -----
# filter to theme within pivot table
theme_series = hap_pivot_table.loc[theme]
# transpose and reset index
theme_series_trans = theme_series.reset_index()
# rename columns
theme_series_trans.columns = ['year_month', 'value']
# ensure 'year_month' is a string before conversion
theme_series_trans['year_month'] = theme_series_trans['year_month'].astype(str)
# convert 'year_month' to datetime format (assuming YYYY-MM format)
theme_series_trans['year_month'] = pd.to_datetime(theme_series_trans['year_month'], format='%Y-%m')
# ensure 'value' is numeric
theme_series_trans['value'] = pd.to_numeric(theme_series_trans['value'], errors='coerce').round(2)
# create inset axes
sub_ax = inset_axes(ax, width="90%", height="70%", loc='center', borderpad=1)

## ----- Smooth the line using interpolation ----
x = theme_series_trans['year_month'].map(lambda d: d.timestamp())  # Convert datetime to numerical timestamps
y = theme_series_trans['value']
# Create interpolation function
x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate more points for smooth curve

spline_viz = make_interp_spline(x, y, k=3)  # k=3 for cubic spline smoothing
spline = UnivariateSpline(x, y, k=3, s=len(x)*1.0)  # s controls the amount of smoothing

y_smooth_viz = spline_viz(x_smooth)
y_smooth = spline(x_smooth)

# === R² CALCULATION ===
y_pred_at_x = spline(x)  # predict at original x
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred_at_x) ** 2)
r_squared = 1 - (ss_res / ss_total)

# Convert timestamps back to datetime for plotting
x_smooth_dates = pd.to_datetime(x_smooth, unit='s')
#fix y smooth
y_smooth[y_smooth < 0] = 0 
# plot the smooth line chart
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='white', linewidth=9, zorder=5)
sub_ax.plot(x_smooth_dates, y_smooth_viz, color='#FFC000', linewidth=2.5, zorder=5)
# remove background from line chart
sub_ax.set_axis_off()

## ----- Display y-axis labels 
max_rate = 35
min_rate = 25
mid_rate = 30
sub_ax.axhline(y=max_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=min_rate, color='black', linestyle='--', lw=0.4)
sub_ax.axhline(y=mid_rate, color='black', linestyle='--', lw=0.4)
sub_ax.text(17350, max_rate - 0.2, f"{max_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold' )
sub_ax.text(17350, min_rate - 0.2, f"{min_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')
sub_ax.text(17350, mid_rate - 0.2, f"{mid_rate:.0f}%",
    fontsize=10, color='black', fontweight='bold')


# ----- set title
lower_bound = 1960
upper_bound = 2019
title = theme.title()
title = title.replace("And","and")

fig_text(
    0.18, 0.817,
    title, color='black',
    fontweight='bold',
    fontsize=16,
    #highlight_textprops=[{"color": 'darkred'},{"fontsize": 13,"color": 'darkgrey',"fontweight": 'normal'}],
    ha='center')

##### ICON, TEXT, ARROW 2 (cost of living payment) -------------------------------------------------------------------------------------------

## ----- Display icon image(s)
# define function to load and return image
def open_image_local(path_to_image):
    image = Image.open(path_to_image) # Open the image
    image_array = np.array(image) # Convert to a numpy array
    return image_array # Output
# initialise image object from file
ld_nov_image = open_image_local('E:/twitter_analysis_data/viz/theme_analysis/icons/hap_cov_open_png.png')
# define the position and size parameters
ld_nov_image_xaxis = 0.45
ld_nov_image_yaxis = 0.595
ld_nov_image_width = 0.08
ld_nov_image_height = 0.08
ax_ld_nov_image = fig.add_axes([ld_nov_image_xaxis, ld_nov_image_yaxis, ld_nov_image_width, ld_nov_image_height])
# show icon
#ax_ld_nov_image.imshow(ld_nov_image)
ax_ld_nov_image.axis('off')  # Remove axis of the image

## ----- Icon text
sub_ax.text(
    0.5, 0.93,  # X, Y in Axes coordinates
    r"$\bf{May\ 2020}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

sub_ax.text(
    0.799, 0.96,  # X, Y in Axes coordinates
    r"$\bf{Jan\ 2022}$" + "\n\n.",
    transform=sub_ax.transAxes,
    color='black',
    fontsize=10,
    #fontweight='bold',
    ha='right',
    va='center')

## ----- Draw arrow
def draw_arrow(tail_position, head_position, invert=True):
    kw = dict(
        arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="k")
    if invert:
        connectionstyle = "arc3,rad=-.5"
    else:
        connectionstyle = "arc3,rad=.5"
    a = FancyArrowPatch(tail_position, head_position,
                        connectionstyle=connectionstyle,
                        transform=fig.transFigure,
                        **kw)
    fig.patches.append(a)
#draw_arrow((0.52, 0.62), (0.58, 0.57))

# --------------------------------------------------------------------------------------------------
## ------ Save figure
save_path = 'E:/twitter_analysis_data/viz/theme_analysis/wc_line_detail/'
plt.savefig(save_path + "hap " + theme + '.png', bbox_inches='tight', pad_inches=0.15)

## ------ Show figure 
plt.show()

################################################# S T A T I S T I C A L   A N A L Y S I S ##########################################

p1 = "Pre-March'20"
p2 = "Post-March'20"

stat_analysis_for_theme(
    anx_or_hap = "Happiness",
    input_dataframe=hap_stat_sig_df,
    theme=theme,
    periods=[
        (p1, "2018-01", "2020-03"),
        (p2, "2020-03", "2023-03")
    ],
    logit_text_positions={
        p1: (0.5, 0.3),
        p2: (0.655, 0.3)
    },
    gee_text_positions={
        p1: 0.05,
        p2: 0.05
    },
    line_color_data='#FFC000',
    r_p_pos = (0.98, 0.95)
)


# # External informational sources for cross-validation
# ---

# ### ONS Personal Well-being estimates
# ---
# https://www.ons.gov.uk/datasets/wellbeing-local-authority/editions/time-series/versions/4

# ### Covid Crisis:
# 
# Covid Timeline: https://www.instituteforgovernment.org.uk/sites/default/files/timeline-lockdown-web.pdf

# ### Cost of Living Crisis:
# 
# Wikipedia: https://en.wikipedia.org/wiki/2021%E2%80%93present_United_Kingdom_cost-of-living_crisis
# 
# Based on an Office for National Statistics (ONS) survey performed between 27 April and 22 May 2022, 77% of UK adults reported feeling worried about the rising cost of living, with 50% saying they worried "nearly every day".
# 
# Cost-of-Living Crisis (2022–2023): Rising inflation and economic challenges led to a cost-of-living crisis. By January 2023, approximately 93% of adults identified the cost of living as a major concern. 
# ONS: https://www.ons.gov.uk/peoplepopulationandcommunity/wellbeing/bulletins/publicopinionsandsocialtrendsgreatbritain/11to22january2023?utm_source=chatgpt.com
#  This period also saw increased public discourse on poverty and economic inequality. 
# NATIONAL CENTRE FOR SOCIAL RESEARCH: https://natcen.ac.uk/news/change-public-mood-creates-challenge-next-government?utm_source=chatgpt.com

# ### Russia-Ukraine War:
# 
# House of Commons Timeline: https://researchbriefings.files.parliament.uk/documents/CBP-9476/CBP-9476.pdf
# 
# Key events
# 13 November 2021: President Zelenskyy says nearly 100,000 Russian troops have massed on the border with Ukraine.
# 17 December 2021: Russia presents a list of security demands in order to defuse the crisis over Ukraine, including a legally binding guarantee that Ukraine will never be accepted as a NATO Member State and that NATO will give up any military activity in eastern Europe and Ukraine.
# 22 January 2022: In a rare reference to intelligence gathering, the Foreign Office exposes evidence of a plot to install a pro-Russian government in Ukraine.
# 24 January 2022: The US places 8,500 troops on heightened alert to deploy to Europe as NATO reinforces its eastern borders with warships and fighter jets.
# 10 February 2022: Russia launches what is being called its largest military exercise since the Cold War, holding joint manoeuvres with Belarus, close to the Belarus/Ukrainian border.
# 21 February 2022: President Putin recognises the independence of the self-proclaimed Donetsk People’s Republic and the Luhansk People’s Republic. He then orders Russian troops into the territories for what he describes as “peacekeeping duties”.
# 
# Dates for the Homes for Ukraine program
# March 2022: The Homes for Ukraine program opened 
# February 19, 2024: The Ukraine Family Scheme closed to new applications 
# May 16, 2024: The Ukraine Extension Scheme closed for most new applications 
# February 4, 2025: Holders of the Ukraine Permission Extension scheme will be able to apply to renew their visas for another 18 months 
# 

# ### National Politics:
# 
# December 12, 2019 – A general election was held, resulting in a Conservative Party victory under Boris Johnson, securing an 80-seat majority in the House of Commons.
# 
# Political Instability and Leadership Changes (2022): The resignation of Prime Minister Liz Truss in October 2022, after a brief tenure marked by economic challenges, contributed to political uncertainty. Her successor faced the task of stabilizing the government amidst public skepticism.

# ### National Center for Social Research 
# Change of public mood creates challenge for the next government
# https://natcen.ac.uk/news/change-public-mood-creates-challenge-next-government?utm_source=chatgpt.com
