#!/usr/bin/env python
# coding: utf-8

# # LLM Call per User-Month Tweet Block
# ---
# The Python code below:
#  - loads and merges csv files containing user-month tweet data
#  - samples user months down to a volume that may be contained within the GPT4o-mini context window 
#  - constructs a prompt to make inferences as to the well-being of resident-users
#  - parses completion data to JSON format
#  - stores completions locally as JSON files

# ## Import Libraries
# ---

# In[ ]:


#inport libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import ast
import json
import tiktoken
from pydantic import BaseModel
import openai
from openai import OpenAI


# In[ ]:





# ## Load full dataset, user month tweet blocks modelled for prompt input
# ---

# In[ ]:


#defining a function to create a single pandas data frame from all csvs within a specified file path
def merge_csv_files(folder_path):
    """
    Merge all CSV files in a folder into a single Pandas DataFrame.
    Parameters: folder_path (str): Path to the folder containing CSV files.
    Returns: pd.DataFrame: A DataFrame containing data from all the CSV files.
    """
    # List to hold data from each file
    all_data = []
    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file into a DataFrame and append to the list
            df = pd.read_csv(file_path)
            all_data.append(df)
    # Concatenate all dataframes in the list
    merged_data = pd.concat(all_data, ignore_index=True)
    return merged_data


# In[ ]:


base_data_path = "E:/twitter_data/may_23_usr_extract/prompt_input_data/cam_loc_usr_prompt_input_3"


# In[ ]:


usr_mnth_df = merge_csv_files(base_data_path)


# In[ ]:


## drop columns / retain only desired columns
usr_mnth_df = usr_mnth_df[['concat_key', 'usr_mnth_prompt_input_tup']]


# In[ ]:


# Ensure usr_mnth_df is a copy, not a slice
usr_mnth_df = usr_mnth_df.copy()
# Split 'concat_key' and assign directly
split_cols = usr_mnth_df['concat_key'].str.split('|', expand=True)
split_cols.columns = ['year', 'month_num', 'user_id']
usr_mnth_df.loc[:, 'year'] = pd.to_numeric(split_cols['year'], errors='coerce')
usr_mnth_df.loc[:, 'month_num'] = pd.to_numeric(split_cols['month_num'], errors='coerce').fillna(0).astype(int)
usr_mnth_df.loc[:, 'user_id'] = split_cols['user_id'].astype(str)
# Map month abbreviations and names
month_abbrs = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
usr_mnth_df.loc[:, 'month_abbr'] = usr_mnth_df['month_num'].map(month_abbrs)
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
              7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
usr_mnth_df.loc[:, 'month_name'] = usr_mnth_df['month_num'].map(month_names)
# create mnth_year_txt variable
usr_mnth_df['mnth_yr_text'] = usr_mnth_df['month_name'] + " " + usr_mnth_df['year'].astype(str)
# Reorder columns directly
usr_mnth_df = usr_mnth_df[['concat_key','user_id', 'year', 'month_num', 'month_abbr', 'month_name', 'mnth_yr_text', 'usr_mnth_prompt_input_tup']]


# In[ ]:


## Filter any rows from before 2018
usr_mnth_df = usr_mnth_df[usr_mnth_df['year'] > 2017]


# In[ ]:


## Derive list formatted prompt input
usr_mnth_df = usr_mnth_df.rename(columns={'usr_mnth_prompt_input_tup': 'usr_mnth_prompt_input_str'})
usr_mnth_df['usr_mnth_prompt_input_tup_list'] = usr_mnth_df['usr_mnth_prompt_input_str'].apply(ast.literal_eval)


# In[ ]:


# calculate the length of prompt input
usr_mnth_df['prompt_input_str_len'] = usr_mnth_df['usr_mnth_prompt_input_str'].apply(len)
# count of tweets in user/month block (tuples in list)
usr_mnth_df['usr_mnth_tweet_count'] = usr_mnth_df['usr_mnth_prompt_input_tup_list'].apply(len)


# In[ ]:


# count tokens per input 
encoding = tiktoken.get_encoding("o200k_base")
def o200k_tokens_from_string(string:str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens
usr_mnth_df['prompt_str_o200k_token_count'] = usr_mnth_df['usr_mnth_prompt_input_str'].apply(o200k_tokens_from_string)
## calaculat etotal input tokens including month value (4) system content (40) and user content(3079)
usr_mnth_df['total_input_tokens'] = usr_mnth_df['prompt_str_o200k_token_count'] + 4 + 40 + 3079  


# ## Summary Statistics (source user-month tweet dataset)
# ---

# In[ ]:


## summary stats:
print("SUMMARY STATISTICS:")
# total user/month blocks
print("Total user / month blocks: " + str(len(usr_mnth_df['concat_key'])))
# unique users
print("Total unique users: " + str(usr_mnth_df['user_id'].nunique(dropna=False)))
# total tweets
print("Total tweets: " + str(sum(usr_mnth_df['usr_mnth_tweet_count'])))
# total tweet block tokens (with full system and users content messages)
print("Total tweet block tokens: " + str(sum(usr_mnth_df['prompt_str_o200k_token_count'])))
# total input tokens (with full system and users content messages)
print("Total total input tokens: " + str(sum(usr_mnth_df['total_input_tokens'])))


# ## Sample user months with tweet chunks exceeding the threshold allowed for prompt input
# ---

# In[ ]:


print("Highest total input tokens: " + str(max(usr_mnth_df['total_input_tokens'])))


# In[ ]:


threshold = 120000
count = (usr_mnth_df['total_input_tokens'] > threshold).sum()
print("User/mnth chunks with over " + str(threshold) + " total input tokens: " + str(count))


# In[ ]:


## create new df containing all user month rows exceeding threshold tokens
large_tweet_chunks_df = usr_mnth_df[usr_mnth_df['total_input_tokens'] > threshold].sort_values(by='total_input_tokens', ascending=False)


# In[ ]:


large_tweet_chunks_df.shape


# In[ ]:


print("Lowest tweet count amongst large tweet chunks: " + str(large_tweet_chunks_df['usr_mnth_tweet_count'].min()))


# In[ ]:


#explode df heavy user subset on tweet tup list
expld_large_tweet_chunks_df = large_tweet_chunks_df.explode('usr_mnth_prompt_input_tup_list', ignore_index=True)
expld_large_tweet_chunks_df = expld_large_tweet_chunks_df.rename(columns={'usr_mnth_prompt_input_tup_list': 'usr_mnth_prompt_input_tup'})
expld_large_tweet_chunks_df['usr_mnth_prompt_input_tup_str'] = expld_large_tweet_chunks_df['usr_mnth_prompt_input_tup'].astype('str')


# In[ ]:


#apply token counter to string version of tweet tuples 
expld_large_tweet_chunks_df['tweet_tup_token_count'] = expld_large_tweet_chunks_df['usr_mnth_prompt_input_tup_str'].apply(o200k_tokens_from_string)


# In[ ]:


expld_large_tweet_chunks_df.shape


# In[ ]:


expld_large_tweet_chunks_df.sample(3)


# In[ ]:


## sampling process
unique_concat_keys = expld_large_tweet_chunks_df['concat_key'].unique()
threshold = 120000
random_state = 42
iterations = list(range(10, 6001, 10))
sampled_df = pd.DataFrame()
for x in unique_concat_keys:
    # create df for usrmnth subset
    print("\n")
    print(x)
    df = expld_large_tweet_chunks_df[expld_large_tweet_chunks_df['concat_key'] == x]
    print("Total starting tweets: " + str(df['usr_mnth_prompt_input_tup_str'].count()))
    print("Total starting tokens: " + str(df['tweet_tup_token_count'].sum()))
    # set starting sample size
    max_sample = 6000
    if max_sample > int(df['usr_mnth_prompt_input_tup_str'].count()):
        max_sample = int(df['usr_mnth_prompt_input_tup_str'].count())
    else:
        max_sample = max_sample
    print("Starting sample size: " + str(max_sample))

    # set max sample size list to iterate through when sampling usr month subset df
    max_sample_list = [max_sample] 
    for num in iterations:  #list of integers to loop through as possible sample sizes
                    max_sample_list.append(int(max_sample - num))   
                    if max_sample_list[-1] < 20:
                        break
    print("max_sample_list being used for " + str(x) + ": from " + str(max_sample_list[0]) + " to " + str(max_sample_list[-1])) 
    for samp_size in max_sample_list:
        df_sample = df.sample(int(samp_size), random_state=random_state)
        tokens_in_sample = df_sample['tweet_tup_token_count'].sum()
        if tokens_in_sample < threshold:
            print("Sampling loop broken at " + str(samp_size) + " tweets, " + str(df_sample['tweet_tup_token_count'].sum()) + " tokens." )
            break
    #append to dataframe
    sampled_df = pd.concat([sampled_df, df_sample], ignore_index=True) 
    print(str(x) + " stored to main df with a sample size of " + str(samp_size) + " / "
          + str(int(df_sample.shape[0])) + " tweets, equating to " + str(df_sample['tweet_tup_token_count'].sum()) + " tokens.")
print("\n")  
print("********************* SAMPLING PROCESS COMPLETE *******************")
print("********************* RESULTANT DATAFRAME SHAPE: " + str(sampled_df.shape))


# In[ ]:


## condense tweet data back into list of tupels


# In[ ]:


sampled_df.sample(3)


# In[ ]:


print("Unique concat_keys: " + str(len(sampled_df['concat_key'].unique())))


# In[ ]:


## create df arranging selected tweet tuples back into a list per concat_key
condensed_sampled_df = sampled_df.groupby('concat_key')['usr_mnth_prompt_input_tup'].apply(list).reset_index()


# In[ ]:


condensed_sampled_df = condensed_sampled_df.rename(columns={'usr_mnth_prompt_input_tup': 'usr_mnth_prompt_input_tup_list'})


# In[ ]:


condensed_sampled_df.sample(3)


# In[ ]:


condensed_sampled_df.shape


# In[ ]:


## rework df of heavy users we started with 
large_tweet_chunks_df.sample(3)


# In[ ]:


large_tweet_chunks_df.shape


# In[ ]:


large_tweet_chunks_df.columns


# In[ ]:


# merge in new sampled df
reduced_large_tweet_chunks_df = large_tweet_chunks_df.drop(columns=['usr_mnth_prompt_input_str', 'usr_mnth_prompt_input_tup_list',
                                                                    'prompt_input_str_len', 'usr_mnth_tweet_count', 'prompt_str_o200k_token_count',
                                                                    'total_input_tokens'])


# In[ ]:


reduced_large_tweet_chunks_df.shape


# In[ ]:


reduced_large_tweet_chunks_df = pd.merge(reduced_large_tweet_chunks_df, condensed_sampled_df, on='concat_key', how='left')
reduced_large_tweet_chunks_df['usr_mnth_prompt_input_str'] = reduced_large_tweet_chunks_df['usr_mnth_prompt_input_tup_list'].astype('str')
reduced_large_tweet_chunks_df['prompt_input_str_len'] = reduced_large_tweet_chunks_df['usr_mnth_prompt_input_str'].apply(len)
reduced_large_tweet_chunks_df['usr_mnth_tweet_count'] = reduced_large_tweet_chunks_df['usr_mnth_prompt_input_tup_list'].apply(len)
reduced_large_tweet_chunks_df['prompt_str_o200k_token_count'] = reduced_large_tweet_chunks_df['usr_mnth_prompt_input_str'].apply(o200k_tokens_from_string)
reduced_large_tweet_chunks_df['total_input_tokens'] = reduced_large_tweet_chunks_df['prompt_str_o200k_token_count'] + 4 + 40 + 3079


# In[ ]:


reduced_large_tweet_chunks_df.shape


# In[ ]:


reduced_large_tweet_chunks_df.columns


# In[ ]:


col_order = list(usr_mnth_df.columns)
reduced_large_tweet_chunks_df = reduced_large_tweet_chunks_df[col_order]
usr_mths_to_replace = reduced_large_tweet_chunks_df['concat_key'].unique()
usr_mnth_df = usr_mnth_df[~usr_mnth_df['concat_key'].isin(usr_mths_to_replace)]
usr_mnth_df = pd.concat([usr_mnth_df, reduced_large_tweet_chunks_df])


# ## Summary Statistics (sampled user-month tweet dataset)

# In[ ]:


## summary stats:
print("SUMMARY STATISTICS:")
# total user/month blocks
print("Total user / month blocks: " + str(len(usr_mnth_df['concat_key'])))
# unique users
print("Total unique users: " + str(usr_mnth_df['user_id'].nunique(dropna=False)))
# total tweets
print("Total tweets: " + str(sum(usr_mnth_df['usr_mnth_tweet_count'])))
# total tweet block tokens (with full system and users content messages)
print("Total tweet block tokens: " + str(sum(usr_mnth_df['prompt_str_o200k_token_count'])))
# total input tokens (with full system and users content messages)
print("Total total input tokens: " + str(sum(usr_mnth_df['total_input_tokens'])))


# ## API Key
# ---

# In[ ]:


api_key = 'api_key'


# ## Prompt Engineering
# ---
# 

# In[ ]:


# initialising system 
sys_content = """\
You are an expert in hedonic subjective well-being. \
You are skilled at inferring an individual's affective experience from their written communication. \
You also have extensive knowledge of the conventions of communication on Twitter.\
"""


# In[ ]:


def create_user_content(month, tweets):
    user_content = f"""\ 
You will be provided with a sample of tweets written by an individual over the course of the month {month}. \ 
Each tweet will be proceeded with a timestamp indicating when it was written. \
This sample of tweets will be delimited with ''' characters. \
Consider the hedonic subjective well-being of the individual who wrote these tweets. \
Perform the following actions to generate a single RFC8259 compliant JSON object: \

1 - Answer the question: Which spoken languages are the tweet written in?
Provide your answer as a list of spoken languages. \
Your response should be given as a Python list of strings against the following key: "languages". \

2 - Answer the question: How happy was the individual?\
Provide your answer as an integer from 0 to 10, against the following key: "hap_lvl". \
Provide a judgment of the level of confidence that you have in your answer as a float from 0 to 1, against the following key: "hap_conf". \
Provide a list of topics that were contributing to the individual's happiness, against the following key: "hap_topics". \
Provide an explanation of your reasoning as a Python string in no more than 200 tokens, against the following key: "hap_explain". \

3 - Answer the question: How anxious was the individual? \
Provide your answer as an integer from 0 to 10, against the following key: "anx_lvl" \
Provide a judgment of the level of confidence that you have in your answer as a float from 0 to 1, against the following key: "anx_conf". \
Provide a list of topics that were contributing to the individual's anxiety, against the following key: "anx_topics". \
Provide an explanation of your reasoning as a Python string in no more than 200 tokens, against the following key: "anx_explain". \

4 - Answer the question: Which emotions were expressed within the tweets? \
Provide your answer as a list of emotions. \
Your responses should be given as a python list of strings against the following key: "emotions". \

5 - Do the tweets reference any of the topics in the following list?: ['cultural identity', 'local spaces', \
'local services', 'community connections', 'housing', 'employment', 'personal finances', 'local environment', \
'feelings of safety', 'learning opportunities', 'educational outcomes'] \
Your response should be given as a Python list of strings against the following key: "glc_topics".

Return all of your responses together in a single RFC8259 compliant JSON object.

Twitter posts: '''{tweets}''' \
"""
    return user_content


# ## Define functions
# ---

# In[ ]:


# define new pydantic class model 
class LLM_output(BaseModel):
    languages: list[str]
    hap_lvl: int
    hap_conf: float
    hap_topics: list[str]
    hap_explain: str
    anx_lvl: int
    anx_conf: float 
    anx_topics: list[str]
    anx_explain: str
    emotions: list[str]
    glc_topics: list[str]


# In[ ]:


# define new pedantic class model 
class LLM_output(BaseModel):
    languages: list[str]
    hap_lvl: int
    hap_conf: float
    hap_topics: list[str]
    hap_explain: str
    anx_lvl: int
    anx_conf: float 
    anx_topics: list[str]
    anx_explain: str
    emotions: list[str]
    glc_topics: list[str]

# define LLM API call funtion 
def run_llm_call(model, sys_content, user_content):
    model = model, 
    completion = client.beta.chat.completions.parse(model=model,
                                                    messages=[{"role": "system", "content": system_content},
                                                              {"role": "user", "content": user_content}],
                                                    response_format=LLM_output)  #response_format = {"type":"json_object"},may be used as 'JSON mode'
    llm_completion = completion.choices[0].message.parsed
    llm_completion_json = llm_completion.model_dump_json()
    return llm_completion_json


# ## Identifying remaining user months to processes
# ---

# In[ ]:


# specify output path
output_path = 'E:/twitter_analysis_data/llm_outputs/jan_03_full_extraction/'
# list of all distinct user months in the dataset
all_usr_month = usr_mnth_df['concat_key'].unique()
# list of names of files in output path - reformated to pipe (concat key convention)
files_output = os.listdir(output_path)
user_months_completed = []
for f in files_output:
    usr = f.replace(".json","")
    usr = usr.replace("_","|")
    user_months_completed.append(usr)
# remaining unprocessed user/months list (all minus completed)
user_months_left = [x for x in all_usr_month if x not in user_months_completed]


# ## Run LLM call process
# ---

# In[ ]:


#perameters
model = "gpt-4o-mini"
client = OpenAI(api_key=api_key)


# In[ ]:


model = "gpt-4o-mini"    # model selection
client = OpenAI(api_key=api_key)   # initialise client object with API key (stored externally)
# apply LLM call porcesses to each item (concatonate_key string) within user_months_left list
for x in user_months_left:
    print ("\n******** P R O C E S S I N G USER/MONTH: "+ x +" *********************")
    month = list(usr_mnth_df.loc[usr_mnth_df['concat_key'] == x, 'mnth_yr_text'])[0]  # initialise 'month' string, for specified row of df
    tweets = list(usr_mnth_df.loc[usr_mnth_df['concat_key'] == x, 'usr_mnth_prompt_input_str'])[0]  # initials 'tweet'string, for specified row of df
    tweets = tweets.encode('utf-8').decode('utf-8') #esnuring sting is 'utf-8' encodable
    system_content = sys_content  # consitent for all api calls, defined outside of loop 
    user_content = create_user_content(month, tweets) # user_content block created from combining yr_month str and tweet block str with prompt text
    # calculating full input tokens
    full_input_string = system_content + user_content #counting input string
    full_input_string = full_input_string.encode('utf-8').decode('utf-8') # re-asserting 'utf-8' encoding for full prompt input string
    total_input_tokens = o200k_tokens_from_string(full_input_string) # quantify total tokens using pre-degined function, using tiktoken
    print("Total input tokens: " + str(total_input_tokens)) # print total input tokens
    # llm call to specified model
    sys_dict = {"role": "system", "content": system_content} # intialise sys_dict object, dictionary for system content message
    user_dict = {"role": "user", "content": user_content} # intialise sys_dict object, dictionary for user content message including month and tweeets  
    completion = client.beta.chat.completions.parse(model=model,   # model specified above used
                                                    messages=[sys_dict, user_dict],   # list of dicts, applied as messages argument
                                                    temperature=0.0, # temperature set to 0 to ensure reproducibility 
                                                    response_format=LLM_output) # pydantic class used to format output
    llm_completion = completion.choices[0].message.parsed    # initialising llm_completion object as llm output completion string
    llm_completion_json = llm_completion.model_dump_json()   # initialising llm_completion_json as json formated output
    output_file_name = x # out_file_name string object intialised as concatonate key
    output_file_name = output_file_name.replace("|","_") # pipes of concat key replaced with underscores for output json file name
    # dumping json formated llm completion to json file, ecoding as 'uft-8', using file name specified above
    with open(output_path + output_file_name + '.json', 'w', encoding='utf-8') as f:
        json.dump(llm_completion_json, f, ensure_ascii=False)
    print ("******** P R O C E S S  C O M P L E T E JSON FILE STORED FOR: "+ x +" *********************")


# ## Load output JSON files into a data frame for review
# ---

# In[ ]:


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
                        token_count = o200k_tokens_from_string(data_as_str)
                        data["output_token_count"] = token_count
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


folder_path = 'E:/twitter_analysis_data/llm_outputs/dec_31_test_outputs/'
df = load_json_files_to_dataframe(folder_path)


# In[ ]:


archive_file_name = '0915pm311224_4omini'
archive_file_path = 'E:/twitter_analysis_data/llm_outputs/archived test sets/' + archive_file_name
df.to_csv(archive_file_path + ".csv", index=False)

