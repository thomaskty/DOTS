#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 00:14:28 2025

@author: thomas
"""


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Sample text
text = "Running is better than walking! Apples and oranges are different."
# Lowercasing
text_lower = text.lower()
print("Lowercased text:", text_lower)

# Removing punctuation
text_no_punct = re.sub(r'[^\w\s]', '', text_lower)
print("Text without punctuation:", text_no_punct)

# Tokenization
words = nltk.word_tokenize(text_no_punct)
print("Tokenized words:", words)

# Removing stop words
stop_words = set(stopwords.words('english'))
words_no_stop = [word for word in words if word not in stop_words]
print("Text without stopwords:", words_no_stop)

# Stemming
ps = PorterStemmer()
words_stemmed = [ps.stem(word) for word in words_no_stop]
print("Stemmed words:", words_stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
words_lemmatized = [lemmatizer.lemmatize(word) for word in words_no_stop]
print("Lemmatized words:", words_lemmatized)


from nltk import pos_tag
from nltk import word_tokenize

text = "GeeksforGeeks is a Computer Science platform."
tokenized_text = word_tokenize(text)
tags = tokens_tag = pos_tag(tokenized_text)
tags


## BAG OF WORDS 
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
documents = [
    "Barack Obama was the 44th President of the United States",
    "The President lives in the White House",
    "The United States has a strong economy"
]

# Initialize the CountVectorizer
vectorizer = CountVectorizer()

# Fit the model and transform the documents into a Bag of Words
bow_matrix = vectorizer.fit_transform(documents)

# Get the feature names (unique words in the corpus)
feature_names = vectorizer.get_feature_names_out()

# Convert the Bag of Words matrix into an array
bow_array = bow_matrix.toarray()

# Display the Bag of Words
print("Feature Names (Words):", feature_names)
print("\nBag of Words Representation:")
print(bow_array)


# tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data (documents)
documents = [
    "The cat sat on the mat.",
    "The cat sat on the bed.",
    "The dog barked."
]

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the model and transform the documents into TF-IDF representation
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (unique words in the corpus)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix into an array
tfidf_array = tfidf_matrix.toarray()

# Display the TF-IDF matrix
print("Feature Names (Words):", feature_names)
print("\nTF-IDF Matrix:")
print(tfidf_array)



# n - grams 
import nltk
from nltk.util import ngrams
from collections import Counter

# Sample text data
text = "The quick brown fox jumps over the lazy dog"

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Generate Unigrams (1-gram)
unigrams = list(ngrams(tokens, 1))
print("Unigrams:")
print(unigrams)

# Generate Bigrams (2-gram)
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Generate Trigrams (3-gram)
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Count frequency of each n-gram (for demonstration)
unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)
# Print frequencies (optional)
print("\nUnigram Frequencies:")
print(unigram_freq)
print("\nBigram Frequencies:")
print(bigram_freq)
print("\nTrigram Frequencies:")
print(trigram_freq)


# word embeddings 
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Sample text data
text = [
    "The cat sat on the mat",
    "The dog barked at the cat",
    "The cat chased the mouse",
    "The dog chased the cat",
]

# Tokenize the sentences into words
tokenized_text = [word_tokenize(sentence.lower()) for sentence in text]

# Train a Word2Vec model on the tokenized text
model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, sg=0)

# Get the word embeddings for a specific word
cat_vector = model.wv['cat']

# Print the word embedding for 'cat'
print("Word Embedding for 'cat':")
print(cat_vector)

# Find words most similar to 'cat'
similar_words = model.wv.most_similar('cat', topn=3)
print("\nWords most similar to 'cat':")
print(similar_words)



# bert - contextual embeddings 
from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example sentence
sentence = "The cat sat on the mat."

# Tokenize the input sentence and convert tokens to tensor
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Pass the input through the BERT model to get embeddings
with torch.no_grad():
    outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state

# Print the shape of the last hidden states tensor
print("Shape of last hidden states:", last_hidden_states.shape)

# Convert the embeddings to numpy array (for easier manipulation)
embeddings = last_hidden_states.squeeze().numpy()

# Tokenize the sentence to match embeddings to words
tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

# Print the tokens and their corresponding contextual embeddings
for token, embedding in zip(tokens, embeddings):
    print(f"Token: {token}")
    print(f"Embedding: {embedding[:10]}...")  # Print first 10 dimensions for brevity
    print()
    
   
# NER 
import spacy

# Load the pre-trained NLP model from spacy
nlp = spacy.load('en_core_web_sm')

# The sentence for which we want to perform NER
sentence = "Barack Obama was the 44th President of the United States."

# Process the sentence using the NLP model
doc = nlp(sentence)

# Print the named entities recognized in the sentence
print("Named Entities in the sentence:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")