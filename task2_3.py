#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Anandh Sellamuthu
# #### Student ID: S3976934
# 
# Date: 17th Sept, 2023
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
# * matplotlib.pyplot (as plt)
# * seaborn
# * sklearn.datasets
# * gensim.models.fasttext
# * sklearn.feature_extraction.text
# * nltk.probability
# * nltk.RegexpTokenizer
# * nltk.tokenize.sent_tokenize
# * itertools.chain
# * nltk.tokenize.word_tokenize
# * sklearn.model_selection
# * sklearn.linear_model
# * LogisticRegression
# * sklearn.metrics
# * gensim.corpora
# * gensim.models.tfidfmodel
# * gensim.matutils
# 
# ## Introduction
# In these tasks, we'll be using the tokenised descriptions generated in task 1 (along with the vocabulary) to generate feature representations. We'll be training a machine learning model (Logistic Regression) on the TF-IDF weighted vector and the unweighted vector generated. This will also be extended to titles and a combination of both titles and descriptions to compare how the models perform depending on the amount of information provided. Each task is divided into sub-tasks for easier reading.

# ## Importing libraries 

# In[1]:


# Importing necessary libraries
import re  
import nltk
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.datasets import load_files
from gensim.models.fasttext import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import *
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.matutils import sparse2full
from sklearn.model_selection import KFold


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# In this task, feature representations will be generated for the descriptions (both weighted and unweighted vectors). We start off with reading the files to generate the necessary data in appropriate data structures. Then, a count vector representation of the files is generated and exported. The features generated in this task will be used in the next task extensively.

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

# Loading data
jp_data = load_files(r"data", encoding="utf-8")
job_ads = jp_data.data  # List of job advertisements

sorted_filenames = sorted(jp_data.filenames, key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Initialising lists to store webindices, titles, and descriptions
webindices = []
titles = []
descriptions = []
target_indices = []

# Extracting webindices, titles, and descriptions for each job ad
for filename in sorted_filenames:
    with open(filename, 'r', encoding="utf-8") as file:
        webindex, title, description = extract_webindex_title_and_description(file.read())
        
        # Extracting the category name (e.g., 'Healthcare_Nursing') from the filename
        category = os.path.basename(os.path.dirname(filename))
        
        # Finding the index of the category in the target_names list
        target_index = jp_data.target_names.index(category)
        
        webindices.append(webindex)
        titles.append(title)
        descriptions.append(description)
        target_indices.append(target_index)


# In[3]:


#Path to the tokenized_descriptions.txt
tkD_filepath = 'tokenized_descriptions.txt'
# Lines from the file will be stored in this list
tokenized_descriptions = []
# Reading file content to the list
with open(tkD_filepath, 'r', encoding='utf-8') as file:
    for line in file:
        tokenized_descriptions.append(line.strip())


# In[4]:


vocab_file_path = 'vocab.txt'
# Initialising list to store the vocabulary read from vocab.txt
vocabulary = []
with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
    for line in vocab_file:
        # Spliting each line using ':' as the delimiter and keep the first part (the word)
        word = line.strip().split(':')[0]        
        # Appending the word to the vocabulary list
        vocabulary.append(word)


# In[5]:


def validator(data_features, vocab, index, webindices,titles,tokenized_descriptions):
    print("Web Index:", webindices[index]) #Printing the corresponding WebIndex for verification
    print("--------------------------------------------")
    print("Title:",titles[index]) #Printing the corresponding Title for verification
    print("Description tokens:",tokenized_descriptions[index]) #Printing the corresponding tokenised description for verification
    print("--------------------------------------------")
    print("Vector representation:\n") #Printing the vector representation

    for word, value in zip(vocab, data_features.toarray()[index]): 
        if value > 0:
            print(word+":"+str(value), end =' ')


# ### Task 2.1 Generating Count Vector (Bag-Of-Words Model)
# Count Vector is a useful feature when it comes to NLP, we generate a count vector, which is essentially a matrix that contains the counts of each word present in a document, on a document-by-document basis. This is exported as a text file below (albeit won't be used for training the models further down the line)

# In[6]:


# Initialise the CountVectorizer with the provided vocabulary
cVectorizer = CountVectorizer(analyzer="word", vocabulary=vocabulary)
# Generate count vectors for tokenized descriptions
count_features = cVectorizer.transform(tokenized_descriptions)

def generate_count_vectors_and_webindices(webindices, vocabulary):

    # Initialize a list to store the results
    results = []

    # Iterate through the webindices and corresponding count vectors
    for webindex, count_vector in zip(webindices, count_features):
        # Create a list of "word_count:count" for non-zero entries in the count vector
        count_vector_str = [f"{ind}:{val}" for ind, val in zip(count_vector.indices, count_vector.data)]

        # Combine webindex, count vector, and format it as a string
        result_str = f"#{webindex}, {' '.join(count_vector_str)}"

        # Append the result string to the list
        results.append(result_str)

    return results


# ### Task 2.2: Generating feature vector representation | FastText Model
# For this task, FastText Model has been opted as it is known to train faster than Word2Vec and is also known to work better for out-of-vocabulary words. We start off with reading the tokenised descriptions file in the appropriate format. The FastText model is trained on this text. A vocabulary is built by the model itself. By extracting the word vectors/keyed vectors from the model, we can proceed with generating weighted and unweighted vectors.

# In[7]:


txt_fname = 'tokenized_descriptions.txt'
with open(txt_fname) as txtf:
    desc_texts = txtf.read().splitlines() 
tk_desc = [a.split(' ') for a in desc_texts]


# In[8]:


# Path to the tokenized corpus file
corpus_file = './tokenized_descriptions.txt'

# Initialising FastText model with vector size of 100
jpFastText = FastText(vector_size=100) 

# Building the vocabulary from the corpus file
jpFastText.build_vocab(corpus_file=corpus_file)

# Training the FastText model on the corpus data
jpFastText.train(
    corpus_file=corpus_file, epochs=jpFastText.epochs,
    total_examples=jpFastText.corpus_count, total_words=jpFastText.corpus_total_words,
)

print(jpFastText)


# In[9]:


# Extracting the word vectors from the trained FastText model
jpFastText_wv = jpFastText.wv


# #### Generating TF-IDF weighted vector using FastText

# In[10]:


# Creating a dictionary mapping between words and their integer ids
descDict = Dictionary(tk_desc) 
descDict.compactify()


# In[11]:


# Converting the tokenized descriptions to bag-of-words using the dictionary created above
descCorpus = [descDict.doc2bow(doc) for doc in tk_desc]

# Initialising the TF-IDF model using the bag-of-words corpus and dictionary and calculating the TF-IDF weights for the documents
model_tfidf = TfidfModel(descCorpus, id2word=descDict)
desc_tfidf  = model_tfidf[descCorpus]


# In[12]:


# Converting the sparse TF-IDF vectors to dense vectors
descriptionVectors = np.vstack([sparse2full(c,len(descDict)) for c in desc_tfidf])


# #### Generating unweighted vectors using FastText

# In[13]:


def unweightedVectors(embeddings, text):
    # Initialise an array to store unweighted vectors for each document
    vecs = np.zeros((len(text), embeddings.vector_size))
    for i, doc in enumerate(text):
        # Filter out terms that are not present in the embeddings
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        if not valid_keys: # Assign a zero vector if no valid terms are found
            docvec = np.zeros(embeddings.vector_size)
        else: # Get embeddings for each valid term and stack them vertically
            docvec = np.vstack([embeddings[term] for term in valid_keys])
            docvec = np.sum(docvec, axis=0) # Sum embeddings row-wise to get unweighted vectors for the documents
        # Store the unweighted vector for the current document in the 'vecs' array
        vecs[i,:] = docvec
    return vecs


# In[14]:


# Generating unweighted vectors for the descriptions
jpFT_unweighted = unweightedVectors(jpFastText_wv, tk_desc)


# ### Saving outputs

# In[15]:


#Verifying if the vectors are correctly generated for each job posting/tokenised description
validator(count_features, vocabulary, 4, webindices,titles,tokenized_descriptions)


# In[16]:


# Generating count vectors
count_vectors_and_webindices = generate_count_vectors_and_webindices(webindices, vocabulary)

# Saving the results to a file
with open("count_vectors.txt", "w", encoding="utf-8") as file:
    for line in count_vectors_and_webindices:
        file.write(line + "\n")


# Upon manual verification (comparing the indices in the vector representation and the vocabulary file), the counts do appear to be correct. 

# ## Task 3. Job Advertisement Classification

# In this task, as alluded to in the introduction, we'll be classifying the job advertisements using the feature vectors we've created in the previous task. Task 3.1 involves the comparison of the vectors created specifically for the descriptions. The other two scenarios (titles only and title + descriptions) will be compared in Task 3.2. 

# ### Task 3.1: Language Model Comparisons
# 
# A comparison of algorithms such as Logistic Regression and LightGBM was performed externally. Logistic Regression was picked for these tasks as it significantly outperforms LightGBM.

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(
    descriptionVectors,  # Training the model using the features generated
    target_indices,  # Target labels (class indices)
    test_size=0.2,  # Test size is assigned to 20% of the data
    random_state=0  
)

logistic_reg = LogisticRegression(random_state=0, max_iter=1000)  # Initialising the Logistic Regression Model
logistic_reg.fit(X_train, y_train)

y_pred = logistic_reg.predict(X_test)

# Generating metrics for the trained model based on the tests and predictions
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# In[18]:


# Generating HeatMap/Visualisation for analysis
categories = ['Accounting/Finance','Engineering','Healthcare/Nursing', 'Sales'] # this gives sorted set of unique label names

sns.heatmap(confusion_mat, annot=True, fmt='d',
            xticklabels=categories, yticklabels=categories) # creates a heatmap from the confusion matrix
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(
    jpFT_unweighted,  # Training the model using the unweighted features
    target_indices,  
    test_size=0.2,  
    random_state=0  
)

logistic_reg = LogisticRegression(random_state=0, max_iter=1000)  
logistic_reg.fit(X_train, y_train)

y_pred = logistic_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# In[20]:


sns.heatmap(confusion_mat, annot=True, fmt='d',
            xticklabels=categories, yticklabels=categories) # creates a heatmap from the confusion matrix
plt.ylabel('Actual')
plt.xlabel('Predicted')


# The metrics convey that the model trained on weighted vectors perform better. We see a prediction accuracy of ~89% with weighted vectors which is 10% more than the score we see for the model trained on unweighted vectors. Further comparisons will be done on other models for a more detailed comparison.

# ### Task 3.2 More information = higher accuracy?

# #### Task 3.2.1: Considering the title only
# Here, we'll be performing only a few of the preprocessing steps that were applied previously for the description as recommended in the discussion forum.

# In[21]:


# Convert titles to lowercase
titles = [title.lower() for title in titles]

#Regex
pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"

# Tokenize the descriptions using the custom pattern
tokenized_titles = [re.findall(pattern, title) for title in titles]


# In[22]:


tokenized_titles = [[t for t in titles if len(t) >=2] \
                      for titles in tokenized_titles]


# In[23]:


filtered_titles = []
for tokenized_title in tokenized_titles:
    filtered_tokens = [token for token in tokenized_title if token in vocabulary]
    filtered_titles.append(filtered_tokens)


# In[24]:


def export(Filename,filtered_titles):
    out_file = open(Filename, 'w') # creates a txt file and open to save the reviews
    string = "\n".join([" ".join(titles) for titles in filtered_titles])
    out_file.write(string)
    out_file.close() # close the file


# In[25]:


export('tokenized_titles.txt',filtered_titles)


# With basic pre-processing done, we can proceed with generating weighted and unweighted vectors for the comparison. Here, the process implemented to generate vectors for the descriptions will be used here as well.

# In[26]:


# Path to the tokenized corpus file
corpus_file = './tokenized_titles.txt'

# Initialising FastText model with vector size of 100
jpTitleFT = FastText(vector_size=100) 

# Building the vocabulary from the corpus file
jpTitleFT.build_vocab(corpus_file=corpus_file)

# Training the FastText model on the corpus data
jpTitleFT.train(
    corpus_file=corpus_file, epochs=jpTitleFT.epochs,
    total_examples=jpTitleFT.corpus_count, total_words=jpTitleFT.corpus_total_words,
)


# In[27]:


jpTitleFT_wv = jpTitleFT.wv # Extracting word vectors from the model


# #### TF-IDF Vector for Titles 

# In[28]:


titles_dict = Dictionary(filtered_titles) 
titles_dict.compactify()


# In[29]:


titles_corpus = [titles_dict.doc2bow(title) for title in filtered_titles] # Converting to Bag of Word
tfidfTitles = TfidfModel(titles_corpus, id2word=titles_dict)
titles_tfidf  = tfidfTitles[titles_corpus]


# In[30]:


titles_vecs = np.vstack([sparse2full(t,len(titles_dict)) for t in titles_tfidf])


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(
    titles_vecs,  # Weighted title feature vectors
    target_indices,  
    test_size=0.2,  
    random_state=0  
)

logistic_reg = LogisticRegression(random_state=0, max_iter=1000)  
logistic_reg.fit(X_train, y_train)

y_pred = logistic_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# #### Unweighted Vector for Titles

# In[32]:


titles_unweighted = unweightedVectors(jpTitleFT_wv, tokenized_titles)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(
    titles_unweighted,  # Unweighted title features
    target_indices,  
    test_size=0.2, 
    random_state=0  
)

logistic_reg = LogisticRegression(random_state=0, max_iter=1000)  
logistic_reg.fit(X_train, y_train)

y_pred = logistic_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)


# Similar to what we can observe when we compare the descriptions, the weighted vectors deliver more accuracy. But we do see a stark difference in accuracy when we compare the TF-IDF trained models. When we train the logistic regression algorithm with the weighted vector generated for the descriptions, we get a model that's ~13% better in terms of accuracy compared to the weighted vector generated for the titles. This is a significant difference. Let's see how these stats compare when we consider both the titles along with their respective descriptions.

# #### Considering Titles + Descriptions
# In this sub-section, the combination of titles and descriptions is used to train our model. There are a few approaches to this but in this scenario, stacking the respective vector representations seems ideal. We'll first verify if the two features are compatible for stacking. If yes, we can proceed with combining the two features and train the model. 

# In[34]:


print('Titles TF-IDF: ', titles_vecs.shape)
print('Descriptions TF-IDF: ', descriptionVectors.shape)
print('----------------------')
print('Titles Unweighted: ', titles_vecs.shape)
print('Descriptions Unweighted: ', descriptionVectors.shape)
print('----------------------')
print('Target length: ', len(target_indices))


# The lengths of each feature and the target are the same, since they are in the same scale, we can proceed with stacking the features.

# In[35]:


X_combined = np.hstack((titles_vecs, descriptionVectors))
# Corresponding labels (y)
y = target_indices
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[36]:


X_unweighted = np.hstack((titles_unweighted, jpFT_unweighted))
# Corresponding labels (y)
y = target_indices
X_train, X_test, y_train, y_test = train_test_split(X_unweighted, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# You may have noticed that the second scenario listed in the question isn't present in this sub-section, that's because it's already been done in Task 3.1 and to avoid redundancy, it hasn't been repeated here. The three scenarios are however compared below using 5-fold cross validation for a more comprehensive comparison.

# ### Comparing the models using 5-Fold Cross Validation

# In[37]:


num_folds = 5
seed = 42
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# In[38]:


def modelEvaluation(X_train,X_test,y_train, y_test,seed):
    model = LogisticRegression(random_state=seed,max_iter = 5000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[39]:


num_models = 6
validationTable = pd.DataFrame(index=range(num_folds))

fold = 0
for train_index, test_index in kf.split(list(range(0, len(target_indices)))):
    y_train = [str(target_indices[i]) for i in train_index]
    y_test = [str(target_indices[i]) for i in test_index]

    #Descriptions
    X_train_tfidf, X_test_tfidf = descriptionVectors[train_index], descriptionVectors[test_index]
    validationTable.loc[fold, 'TF-IDF for Descriptions'] = modelEvaluation(descriptionVectors[train_index], descriptionVectors[test_index], y_train, y_test, seed)
    
    X_train_unweighted, X_test_unweighted = jpFT_unweighted[train_index], jpFT_unweighted[test_index]
    validationTable.loc[fold, 'Unweighted Vector for Descriptions'] = modelEvaluation(X_train_unweighted, X_test_unweighted, y_train, y_test, seed)
    
    #Titles
    X_train_titleIDF, X_test_titleIDF = titles_vecs[train_index], titles_vecs[test_index]
    validationTable.loc[fold, 'TF-IDF for Titles'] = modelEvaluation(titles_vecs[train_index], titles_vecs[test_index], y_train, y_test, seed)
    
    X_train_titleUW, X_test_UW = titles_unweighted[train_index], titles_unweighted[test_index]
    validationTable.loc[fold, 'Unweighted Vector for Titles'] = modelEvaluation(X_train_titleUW, X_test_UW, y_train, y_test, seed)
    
    #Combined
    X_train_CIDF, X_test_CIDF = X_combined[train_index], X_combined[test_index]
    validationTable.loc[fold, 'TF-IDF for Combined Text Tokens'] = modelEvaluation(X_combined[train_index], X_combined[test_index], y_train, y_test, seed)
    
    X_train_CUW, X_test_CUW = X_unweighted[train_index], X_unweighted[test_index]
    validationTable.loc[fold, 'Unweighted Vector for Combined Text Tokens'] = modelEvaluation(X_train_CUW, X_test_CUW, y_train, y_test, seed)

    fold += 1


# In[40]:


validationTable


# In[41]:


print('Best Case Results for each model: ')
max_values = validationTable.max(axis=0)
print(max_values)


# Judging by the results we've obtained, TF-IDF features generally outperform unweighted vectors across all text data types. Combining text tokens from both titles and descriptions (TF-IDF) yields the highest accuracy.
# Unweighted vectors for titles and combined text tokens perform relatively poorly, indicating that word embeddings may not capture semantic information effectively.
# TF-IDF for descriptions and combined text tokens have higher average accuracies compared to other approaches.
# These results suggest that using TF-IDF features for both descriptions and combined text tokens is an effective choice for this classification task.

# ## Summary
# In these tasks, the FastText word embedding model was used to generate TF-IDF vectors for all the components of the corpus (i.e., titles and descriptions). A count_vector.txt file was also generated for the descriptions. Using the feature representations generated, an elaborate comparison was performed on Logistic Regression models trained on various scenarios (as listed in Task 3's specifications) using 5-Fold Cross Validation. The conclusion is that we can expect more accuracy when we use weighted features to train our models. Overall, this has been a challenging but very interesting assignment.

# ## References
# 
# Deep, A 2020, ‘Word2Vec, GLOVE, FastText and Baseline Word Embeddings step by step’, Analytics Vidhya.
# 
# Ganesan, K 2021, ‘FastText vs. Word2vec: A Quick Comparison’, Kavita Ganesan, PhD.
# 
# ‘Introduction to word embeddings – Word2Vec, Glove, FastText and ELMo – Data Science, Machine Learning, Deep Learning’.
# 
# nicogen 2017, ‘Answer to “How to combine tfidf features with selfmade features”’, Stack Overflow.
# 
# Reveille 2020, ‘Answer to “Why concatenate features in machine learning?”’, Stack Overflow.
# 
# Acknowledgement: Some of the codes have been repurposed from the lab and activity notebooks provided on Canvas.
