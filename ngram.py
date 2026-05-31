# %% [markdown]
# # Week 3 Assignment: Implementing N-grams with NLTK
# 
# **Task:**
# Create N-grams for various N across multiple sentences using at least 1 or 2 sentences to apply multiple N values.
# ***

# %% [markdown]
# # Environment Setup & Library Install

# %%
import pandas as pd
import nltk
import spacy

# %% [markdown]
# ***
# # Dataset Collection
# - Select a sample text with at least one or two sentences. 
#     - *Sample text collected from: https://reedsy.com/short-stories/*
# 

# %%
text_segments = {
    "Segment 1 - The Funeral and the Feast": """
    The old woman in a white lace dress had appeared in Claudia’s dream three 
    days before her grandfather died. The dress was so crisp, so white, so 
    beautiful it actually frightened her – not the kind of fear that makes you 
    want to pull the blanket over your head, but the kind that roots you to the 
    spot and freezes the blood coursing through your veins. She had woken from 
    this dream with certainty. She knew what was coming. And she had been right.
    """,

    "Segment 2 - The Funeral and the Feast": """
    Claudia noticed her mother and said nothing. She had spent the whole day calculating 
    the right gesture – a hand on the arm, a quiet word of comfort – and kept coming up empty.
    For anyone else in mourning, she would have known exactly what to do and say. For her mother, 
    the wires crossed somewhere. They seemed to always be reaching for each other but missing the 
    mark by an inch.    
    """,

    "Segment 3 - The Funeral and the Feast": """
    On her left sat ninety three years old Victor, who had been friends with her 
    grandfather for longer than Claudia’s parents had been alive. He was the spitting 
    image of French movie star from the ‘80s and ‘90s, the beautiful Alain Delon. 
    The man was over six feet tall, full head of silver hair, cleanly shaven and looking 
    dapper in his leather jacket and suit pants. It was easy for Claudia to assume that 
    her grandfather and this man were quite the Casanovas back in the day.    
    """
}

# %% [markdown]
# ***
# # Tokenization
# - Tokenize the selected text into sentences and words using NLTK’s tokenizer.

# %%
# Convert to lowercase; Remove punctuation; Split into sentences and words

def preprocess_text(text):
    #Split into sentences
    sentences = nltk.sent_tokenize(text)

    # Convert to lowercase
    cleaned_text = text.lower()
    
    # Remove punctuation
    cleaned_text = ''.join(
        char for char in cleaned_text 
        if char.isalnum() or char.isspace()
    )
    
    # Split into words
    words = nltk.word_tokenize(cleaned_text)
    
    return sentences, words

# %% [markdown]
# ***
# # Generate N-grams
# - Create N-grams for various values of N (e.g., 1, 2, 3)

# %%
# N-grams for various n values (1, 2, 3)

def generate_ngrams(words, n): # Generate n-grams for a list of words
    ngrams = list(nltk.ngrams(words, n)) # Generate n-grams using nltk
    return ngrams # Return the list of n-grams


# %%
# Extract n-grams for each segment
ngram_results = {}
for segment_name, text in text_segments.items():
    sentences, words = preprocess_text(text)
    ngram_results[segment_name] = {
        'unigrams': generate_ngrams(words, 1),
        'bigrams': generate_ngrams(words, 2),
        'trigrams': generate_ngrams(words, 3)
    }

# %%
# Show All n-grams for Each Segment
for segment_name, ngrams in ngram_results.items():
    print(f"{segment_name} - Unigrams:")
    print(ngrams['unigrams'])
    print(f"{segment_name} - Bigrams:")
    print(ngrams['bigrams'])
    print(f"{segment_name} - Trigrams:")
    print(ngrams['trigrams'])
    print("\n")


# %% [markdown]
# ***
# Jen Allen<br>
# CST 645 100 Natural Language Processing<br>
# Week 3 Assignment<br>
# 05/31/2026<br>
# <br>
# **GitHub:** https://github.com/JeaneeAllen/N-grams_NLTK


