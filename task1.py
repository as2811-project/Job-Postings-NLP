#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Anandh Sellamuthu
# #### Student ID: S3976934
# 
# Date: 14th Sept, 2023
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * re
# * nltk
# * os
# * numpy (as np)
# * pandas (as pd)
# * nltk.probability
# * nltk.RegexpTokenizer
# * nltk.tokenize.sent_tokenize
# * itertools.chain
# * nltk.tokenize.word_tokenize
# 
# ## Introduction
# In this task, we'll be pre-processing the given files. Pre-processing includes tasks such as tokenizing, removing single character tokens, removing tokens based on various statistics (top 50 words in terms of occurrence, for instance). This task covers only the descriptions present in each job posting file. Post-completion, the necessary output file is exported along with tokenized descriptions which will be used for tasks 2 and 3.

# ## Importing libraries 

# In[1]:


#Importing necessary libraries
import re
import nltk
import os
import numpy as np
from sklearn.datasets import load_files
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from nltk.probability import *
from collections import Counter


# ### Examining and loading data

# In[2]:


def extract_webindex_title_and_description(file_content):
    webindex = None
    title = None
    description = None

    lines = file_content.split('\n')  # Decode the bytes to a string
    in_description = False

    for line in lines:
        if line.startswith("Webindex:"):
            webindex = line.split(":")[1].strip()
            in_description = False
        elif line.startswith("Title:"):
            title = line.split(":")[1].strip()
        elif line.startswith("Description:") or in_description:
            if description is None:
                description = line.lstrip("Description:").strip()
                in_description = True
            else:
                description += " " + line.strip()
    
    return webindex, title, description

# Load data
jp_data = load_files(r"data", encoding="utf-8")
job_ads = jp_data.data  # List of job advertisements

#Sorting filenames in order to ensure the order in which the files are read and the tokens are appended to the lists will remain the same.
sorted_filenames = sorted(jp_data.filenames, key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Initialize lists to store webindices, titles, and descriptions
webindices = []
titles = []
descriptions = []

# Extract webindices, titles, and descriptions for each job ad
for filename in sorted_filenames:
    with open(filename, 'r', encoding="utf-8") as file:
        webindex, title, description = extract_webindex_title_and_description(file.read())
        webindices.append(webindex)
        titles.append(title)
        descriptions.append(description)


# In[3]:


jp_data['target_names']


# In[4]:


#Fetching a random filename and its corresponding target/category for verification
jp_data['filenames'][2], jp_data['target'][2]


# The files are loaded and the information that we'll be using for this task are stored in-memory within lists. This includes the Web Index values present in each file along with the titles and descriptions. There are four categories of job postings, namely, Accounting_Finance, Engineering, Healthcare_Nursing, and Sales. The files are read in a sorted order (based on file names). 

# ### Pre-processing data
# In this section, all the pre-processing steps mentioned in the introduction is performed. Each pre-processing step is within its own sub-section for ease of reading.

# In[5]:


# Convert all descriptions to lowercase
descriptions = [description.lower() for description in descriptions]

#Regex
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"

# Tokenize the descriptions using the custom pattern
tokenized_descriptions = [re.findall(pattern, description) for description in descriptions]


# In[6]:


def stats_print(tokenized_descriptions):
    words = list(chain.from_iterable(tokenized_descriptions)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of descriptions:", len(tokenized_descriptions))


# In[7]:


print("Raw jobs:\n",descriptions[0],'\n')
print("Tokenized descriptions:\n",tokenized_descriptions[0])


# In[8]:


stats_print(tokenized_descriptions)


# The next few sub-tasks focus on removing words based on statistics. This involves removing single character tokens, stop words and top 50 words that appear most often. The idea behind this process is that these tokens/words don't add much information. It could potentially only add noise and cause performance degredation when it comes to training classifiers (which will be done in Tasks 2 and 3)

# #### 1.1: Removing single character tokens

# In[9]:


tokenized_descriptions = [[d for d in descriptions if len(d) >=2] for descriptions in tokenized_descriptions]


# Only the tokens with lengths greater than or equal to 2 are retained. 

# #### 1.2: Removing Stop Words

# In[10]:


#Reading the given stopwords file into a set
with open('stopwords_en.txt', 'r') as stopwords_file:
    custom_stopwords = set(stopwords_file.read().split())

#Function to return words not in the set created above
def remove_custom_stopwords(tokens):
    return [word for word in tokens if word not in custom_stopwords]

tokenized_descriptions = [remove_custom_stopwords(tokens) for tokens in tokenized_descriptions]
print(tokenized_descriptions[0])


# In[11]:


stats_print(tokenized_descriptions)


# #### 1.3 Removing words that appear only once using term frequency

# In[12]:


words = list(chain.from_iterable(tokenized_descriptions)) # we put all the tokens in the corpus in a single list
vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words


# In[13]:


# Count the frequency of each word in tokenized_descriptions
word_counts = Counter(word for tokens in tokenized_descriptions for word in tokens)

# Get the least common words (sorted by frequency in ascending order)
least_common_words = word_counts.most_common()[-25:]


# In[14]:


# Count the frequency of each word in the list of words
word_counts = Counter(words)

# Find words that appear only once
unique_words = {word for word, count in word_counts.items() if count == 1}

# Remove words that appear only once from the list of words
filtered_words = [word for word in words if word not in unique_words]

# If needed, recreate the vocabulary set from the filtered words
filtered_vocab = set(filtered_words)
len(unique_words)


# In[15]:


def remove_single_frequency_words(tokenized_descriptions):
    # Flatten the list of tokens
    all_tokens = [token for doc_tokens in tokenized_descriptions for token in doc_tokens]
    
    # Count the frequency of each word
    word_counts = Counter(all_tokens)
    
    # Find single-frequency words
    single_frequency_words = set(word for word, count in word_counts.items() if count == 1)
    
    # Remove single-frequency words from each document
    cleaned_jobs = [[token for token in doc_tokens if token not in single_frequency_words] for doc_tokens in tokenized_descriptions]
    
    return cleaned_jobs


# In[16]:


# Removing single frequency words via the function defined above
tokenized_descriptions = remove_single_frequency_words(tokenized_descriptions)


# In[17]:


stats_print(tokenized_descriptions)


# #### 1.4 Removing top 50 most frequent words based on Document Frequency

# In[18]:


words = list(chain.from_iterable([set(job) for job in tokenized_descriptions]))
doc_fd = FreqDist(words)


# In[19]:


#Calculating document frequency for each word
document_frequency = {word: doc_fd.freq(word) for word in doc_fd.keys()}

#Sorting to retrieve top 50 words by frequency
top_50_words_by_df = sorted(document_frequency.items(), key=lambda x: x[1], reverse=True)[:50]

#Extracting said words from the top 50 list
top_50_words = [word for word, _ in top_50_words_by_df]


# In[20]:


# Function to remove top N number of words (by Document Frequency)
def remove_top_words(tokens, top_words):
    return [word for word in tokens if word not in top_words]


# In[21]:


tokenized_descriptions = [remove_top_words(tokens, top_50_words) for tokens in tokenized_descriptions]


# In[22]:


stats_print(tokenized_descriptions)


# We started off with a vocabulary size of 9834. With each pre-processing step, a more compact version of the vocabulary is returned. Post pre-processing, the vocabulary size is 5168. With that, we can go ahead and export the file in the required naming convention/format.

# ## Saving required outputs

# In[23]:


#Creating the sorted vocabulary list
sorted_vocab = sorted(list(set(word for tokens in tokenized_descriptions for word in tokens)))

#Creating a dictionary to map words to integer indices
word_to_index = {word: index for index, word in enumerate(sorted_vocab)}

#Creating the vocab.txt file
with open('vocab.txt', 'w', encoding='utf-8') as vocab_file:
    #Writing each word and its corresponding index to the file
    for word, index in word_to_index.items():
        vocab_file.write(f"{word}:{index}\n")

print("Vocabulary file 'vocab.txt' has been created.")


# In[24]:


def export_jobs(Filename,tokenized_descriptions):
    out_file = open(Filename, 'w') #Opens a file to write the descriptions
    string = "\n".join([" ".join(jobs) for jobs in tokenized_descriptions])
    out_file.write(string)
    out_file.close()


# In[25]:


export_jobs('tokenized_descriptions.txt',tokenized_descriptions)


# ## Summary
# In this task, all the pre-processing steps specified in the specifications for Task 1 were performed. The required output file, vocab.txt, was generated along with the tokenised descriptions which will be used in tasks 2 and 3. This task helped in gaining insights on the dataset provided and paved the way for the upcoming tasks. 

# ## References
# 
# Swast, T 2009, ‘Answer to “Unicode (UTF-8) reading and writing to files in Python”’, Stack Overflow.
# 
# ‘Unicode (UTF-8) reading and writing to files in Python’, <https://www.w3docs.com/snippets/python/unicode-utf-8-reading-and-writing-to-files-in-python.html>.
# 
# Acknowledgement: Some of the codes used here were repurposed from the lab and activity jupyter notebooks provided on Canvas.
