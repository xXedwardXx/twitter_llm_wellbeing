#!/usr/bin/env python
# coding: utf-8

# #  Resident User Tweet Extraction
# ---
# 
# Having idenitfied Twitter users likley to be residents of the borough, the user ids of these users are used to extract all avaible tweets published by those users. For each user, the full archive search endpoint is queried once for each month. Where the user has published a very large volume of tweets within a given month over 50), the avaible tweets are sampled, to ensure a manageable data volume. nb: tweet counts per month per user have already been extracted via the 'counter' endpoint of the Twitter API.

# ### Import Libraries
# ---

# In[1]:


from twarc import Twarc2, expansions
import datetime
import json
import csv
import numpy as np
import random
import pandas as pd
import calendar
from calendar import monthrange

import requests
import time
import itertools
import os


# ### Load User Month Tweet Counts
# ---

# In[2]:


path = 'E:/twitter_data/nov_22_extract/source_data_for_extract/'
usr_mnth_cnts = pd.read_csv(path + 'usr_mnth_cnts_2.csv', usecols=['user_id','year', 'month','year_month','tweet_count'])
usr_mnth_cnts['usr_yr_month_concat'] = usr_mnth_cnts['user_id'].astype('str') + "|" + usr_mnth_cnts['year_month']

#de-duping to leave max
usr_mnth_cnts = usr_mnth_cnts.sort_values(by=['user_id', 'year', 'month', 'tweet_count'])
usr_mnth_cnts = usr_mnth_cnts.drop_duplicates(subset='usr_yr_month_concat', keep='last')


# In[3]:


usr_mnth_cnts['tweet_count'].loc[usr_mnth_cnts['tweet_count'] > 1000].hist()


# In[4]:


usr_mnth_cnts.loc[usr_mnth_cnts['tweet_count'] > 10000]


# In[5]:


#1095017776845070338 
pd.set_option('display.max_rows', None)
usr_mnth_cnts.loc[usr_mnth_cnts['user_id'] == 702181723048640512]


# In[6]:


len(usr_mnth_cnts['usr_yr_month_concat'].unique())


# In[7]:


# band for querey spliting
#Numeric data can be grouped into categories using the pandas' .cut() method

#create lists of dsired bandings and group names; np.inf in numpy's infinite function
ranges = [0.9, 50, np.inf]
group_names = ['1 to 50', '50+']

#create lists of dsired bandings and group names; np.inf in numpy's infinite function
usr_mnth_cnts['group'] = pd.cut(usr_mnth_cnts['tweet_count'], bins=ranges, labels=group_names)


# In[8]:


#filter out user-yeaR-mnths with no tweets
usr_mnth_cnts = usr_mnth_cnts.loc[usr_mnth_cnts['tweet_count'] > 0]


# In[9]:


len(usr_mnth_cnts['usr_yr_month_concat'].unique())


# In[10]:


usr_mnth_cnts.groupby(["group"]).agg({"tweet_count": np.sum, "user_id": pd.Series.nunique})


# ### Set Year_Month list
# ---

# In[11]:


yr_mnt_list = list(usr_mnth_cnts['year_month'].unique())
# remove any year_months already fully extracted (manually amedn list variable)
yr_mnt_list = [x for x in yr_mnt_list if x not in extract_mnt_yrs]
yr_mnt_list


# ### Twarc2 Extraction of Tweets as .jsonl files
# ---

# In[13]:


output_path = 'E:/twitter_data/nov_22_extract/raw_json/'


# In[14]:


# initialise bearer_token variable
# from .txt file containing bearer token
with open(bearer_token_path + 'bearer_token.txt') as f:
    bearer_token = f.readlines()
bearer_token = bearer_token[0]

# initialise Twarc2 client object with bearer token
client = Twarc2(bearer_token=bearer_token)


# In[15]:


for y_m in yr_mnt_list:
    
    ## show y_m
    print('\n')
    print("y_m: " + y_m)
    
    # intialise yr and mnth
    yr =  int(y_m.split("_")[0])
    mnth = int(y_m.split("_")[1])
    print('\n')
    print("yr: " + str(yr))
    print("mnth: " + str(mnth))
    
    
    # source usr_id list for year month
    urs_cnts_df = usr_mnth_cnts[usr_mnth_cnts['year_month'] == y_m]

    print('\n')
    print("unique users in year month " + str(len(urs_cnts_df["user_id"].unique())))
    print("sum tweets in year month " + str( urs_cnts_df["tweet_count"].sum()))
    
    print(urs_cnts_df)
    
    # source usr_id list for year month
    usrs_1_to_50 = list(urs_cnts_df.loc[(urs_cnts_df['year_month'] == y_m) & (urs_cnts_df['group'] == '1 to 50')]['user_id'])
    #ensure values in list are unique
    usrs_1_to_50 = list(set(usrs_1_to_50))
    
    print('\n')
    print("usr ids with 1 to 50 tweets in year month: " + str(len(usrs_1_to_50)))
    print(usrs_1_to_50)
    
    ############################################################
    ## NULLIFY 1 TO 50 LIST FOR TESTINGT OF OVEER 100 LIST ###
    ###########################################################
    #usrs_1_to_50 = []
    
    #usrs_over_50
    usrs_over_50 = list(urs_cnts_df[(urs_cnts_df['year_month'] == y_m) & (urs_cnts_df['group'] == '50+')]['user_id'])
    #ensure values in list are unique
    usrs_over_50 = list(set(usrs_over_50))
    
    print('\n')
    print("usr ids with over 50 tweets in year month: " + str(len(usrs_over_50)))
    print(usrs_over_50)
   
    # derive days in month
    d_no = monthrange(yr,mnth)[1]
    print("\n>>>>>> MONTH: " + str(yr) + " / " +  str(mnth) + " (" + str(d_no) + " days in month)")
    days_in_month_list = list(range(1, d_no+1, 1))
    print(days_in_month_list)

    
    ## QUERY ALL TWEETS IN YEAR_MOTNH FOR USERS WITH 1 TO 100 TWEETS
    
    for usr in usrs_1_to_50:
        
        print("\n>>> NEW USER, in 1 to 50 cohort: " + str(usr))
        
        # Specify the start time in UTC for the time period you want Tweets from
        start_time = datetime.datetime(yr, mnth, 1, 0, 0, 0, 0, datetime.timezone.utc)
        # Specify the end time in UTC for the time period you want Tweets from
        end_time = datetime.datetime(yr, mnth, d_no, 23, 59, 59, 0, datetime.timezone.utc)
                 
        #usr_id = usr
        usr_input = "from:" + str(usr) + " " + "-is:retweet"
        
        query = usr_input
        
        # The search_all method call the full-archive search endpoint to get Tweets based on the query, start and end times
        search_results = client.search_all(query=query, start_time=start_time, end_time=end_time, max_results=100)
        
        print("\n NEW SEARCH: " + usr_input)
        
        # Twarc returns all Tweets for the criteria set above, so we page through the results        
        for page in search_results:

            print("*NEW PAGE OF SEARCH*")
            
            # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
            # so we use expansions.flatten to get all the information in a single JSON

            result = expansions.flatten(page)
            tweets_returned = len(result)
   
            print(">>> Querying: >>> " + "all days in " + str(yr) + "/" + str(mnth) +  " for user: " + usr_input + " Tweet count: " + str(tweets_returned))

            # intialise pseudoranmon id for file to allow for files to be uniquely identifiable
            id = random.randint(100000000000000,999999999999999)
                        
            print("Random file ID: " + str(id))
                    
            ## STORE AS JSON FILES ##
            with open(output_path + str(yr) + '/' + str(mnth) + '/'+ str(id) + '_' + str(yr) + "-" + str(mnth) + '_usr' + str(usr) + '.jsonl', 'w', encoding ='utf8') as f:
                json.dump(page, f, ensure_ascii=True)   
    
    
    ## QUERY SAMPLING TWEETS IN YEAR_MONTH FOR USERS WITH OVER 100 TWEETS
    
    for usr in usrs_over_50:
        
        print("\n>>> NEW USER, in over 100 cohort: " + str(usr))
        
        rand_day_ex_tup_list = []
        
        for x in days_in_month_list:
            rand_ds_so_far = [x[0] for x in rand_day_ex_tup_list]
            choice_list = [x for x in days_in_month_list if x not in rand_ds_so_far]
            rand_day = random.choice(choice_list)
            
            # Specify the start time in UTC for the time period you want Tweets from
            start_time = datetime.datetime(yr, mnth, rand_day, 0, 0, 0, 0, datetime.timezone.utc)
            # Specify the end time in UTC for the time period you want Tweets from
            end_time = datetime.datetime(yr, mnth, rand_day, 23, 59, 59, 0, datetime.timezone.utc)
                 
            usr_input = "from:" + str(usr) + " " + "-is:retweet"
            query = usr_input
                
            # The search_all method call the full-archive search endpoint to get Tweets based on the query, start and end times
            search_results = client.search_all(query=query, start_time=start_time, end_time=end_time, max_results=50)
            
            print("\n NEW SEARCH: " + usr_input)
            
            # Twarc returns all Tweets for the criteria set above, so we page through the results
            for page in search_results:
                
                print("*NEW PAGE OF SEARCH*")
                
                # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
                # so we use expansions.flatten to get all the information in a single JSON

                result = expansions.flatten(page)
                tweets_returned = len(result)
   
                print(">>> Querying: >>> " + "day: " + str(rand_day) + " in " + str(yr) + "/" + str(mnth) +  " for user: " + usr_input + " Tweets: " + str(tweets_returned))
                
                # intialise pseudoranmon id for file to allow for files to be uniquely identifiable
                id = random.randint(100000000000000,999999999999999)
                        
                print("Random file ID: " + str(id))
                    
                ## STORE AS JSON FILES ##
                with open(output_path + str(yr) + '/' + str(mnth) + '/'+ str(id) + '_' + str(yr) + "-" + str(mnth) + '_usr' + str(usr) + '.jsonl', 'w', encoding ='utf8') as f:
                    json.dump(page, f, ensure_ascii=True)
                
                rand_d_count_tup = (rand_day, tweets_returned)
                rand_day_ex_tup_list.append(rand_d_count_tup)
                #print(rand_day_ex_tup_list)
                
            sum_of_twts_in_month = sum([x[1] for x in rand_day_ex_tup_list])
            #print("sum_of_twts_in_month: " + str(sum_of_twts_in_month))

            if sum_of_twts_in_month > 50: # Only retrieve the first two pages (enumerate starts from 0)
                break
        print("Total Tweets for: "  + str(usr) + " in " + str(yr) + "/" + str(mnth) + ": " + str(sum_of_twts_in_month))

