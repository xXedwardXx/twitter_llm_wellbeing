# twitter_llm_wellbeing

Various code blocks constitute a data analytics pipeline extracting tweet data and analysing well-being using an LLM.

The project...

nb: Some degree of data wrangling and iterative batch processing was required to acquire a dataset of tweets published by 'resident users', 
and the interaction with the LLM was carried out as a one-off exercise for the full dataset, hence the pipeline is delineated into multiple applications/files. 

For each file within the repository, the name of the file, a link to the file, and an explanation of the file and its contents are given below:

---
### geospatial_tweet_extraction.py
- https://github.com/xXedwardXx/twitter_llm_wellbeing/blob/main/geospatial_tweet_extraction.py
- python code sourcing geospatial administrative data, re-modelling those data, using re-modelled data to query the Twitter full archive search endpoint, and storing returned tweet data
---
### resident_user_tweet_extraction.py
- https://github.com/xXedwardXx/twitter_llm_wellbeing/blob/main/resident_user_tweet_extraction.py
- Python code extracting tweets for identified resident users, sampling to reduce data to a manageable volume where the resident user has published more than 50 tweets in a month. 
---
### llm_call_per_user_month_tweet_block.py
- x
- the application
 - loads and merges csv files containing user-month tweet data
 - samples user months down to a volume that may be contained within the GPT4o-mini context window 
 - constructs a prompt to make inferences as to the well-being of resident-users
 - parses completion data to JSON format
 - stores completions locally as JSON files
---
### twitter_llm_output_nlp_env.yml
- https://github.com/xXedwardXx/twitter_llm_wellbeing/blob/main/twitter_llm_output_nlp_env.yml
- a .yml file containing the package environment used to develop and execute the ...
---
### onward_nlp_pipeline.py
- https://github.com/xXedwardXx/twitter_llm_wellbeing/blob/main/onward_nlp_pipeline.py
- a .py file containing Python code used to carry out analysis of data output by the LLM in JSON format
- the application:
  - loads data from JSON source files
  - carries out simple EDA and calculates some summary statistics
  - performs pre-processing, filtering based on confidence levels, and cleansing topic data
  - cross-references LLM-derived mean happiness and anxiety levels against published ONS statistics
  - performs month-on-month longitudinal analysis of happiness and anxiety levels
  - word cloud visualisations
  - TF-IDF experimentation (not included in final analysis)
  - statistical modelling of data aggregated to a monthly prevalence measure
  - outputs visualisations
