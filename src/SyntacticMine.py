from config import DTF_REVIEW
import json, os
import numpy as np
import pandas as pd
import pickle as pk

## spacy packages
import spacy
from spacy import displacy

## vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

## load Spacy English pipeline
# download with: python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def load_dataset():
    with open(DTF_REVIEW, 'r') as f:
        lines = f.readlines()
    return [doc.strip() for doc in lines]

def view_sentence(doc):
    ## convert the doc into spacy nlp object
    nlp_doc = nlp(doc)
    for st_id, st in enumerate(nlp_doc.sents):
        print("Sentence {}:".format(st_id+1), st)

def view_dependency(st):
    for tk in st:
        print(f"{tk.dep_} : ({tk.head.text}, {tk.text})", tk.head.pos_)

def view_st_dependency(doc):
    nlp_doc = nlp(doc)
    for st in nlp_doc.sents:
        view_dependency(st)

def main():
    corpus = load_dataset()
    view_st_dependency(corpus[4])

if __name__ == '__main__':
    main()