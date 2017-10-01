import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

wnl = WordNetLemmatizer()

def map_wordnet_pos(treebank_tag):
    """Map the NLTK pos tags onto the pre-trained Treebank tag set"""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    elif treebank_tag.startswith("S"):
        return wordnet.ADJ_SAT
    else:
        return wordnet.NOUN

def clean_text(document):
    """Clean and lemmatize the submitted content"""
    text = [] # list to hold all lemmatised words in this article
    stop_words = set(stopwords.words("english"))  # load stopwords
    # Tokenize words
    words = word_tokenize(document)
    # Remove punctuation
    words = [x for x in words if x not in string.punctuation]
    # Remove single character words
    words = [x for x in words if len(x) > 1]
    # Remove irrelevant words and characters
    stoplist = set("for a of the and to in 's".split()) # Tried with these too; no difference: </p> <p> \\\" \\\n
    words = [x for x in words if x not in stoplist]
    # Remove stopwords
    words = [x for x in words if x not in stop_words]
    # Derive part-of-speech (POS) tags
    words = nltk.pos_tag(words)
    # Lemmatize the cleaned word set
    for word, pos in words:
        tag = map_wordnet_pos(pos) # Call function to map POS tags from Treebank to NLTK
        lemma = wnl.lemmatize(word, tag)
        text.append(lemma) # Add the lemmatized word to the list for this article
    return text