# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys

import nltk

import pandas as pd

from textblob.classifiers import NaiveBayesClassifier

training_data=pd.read_csv('questions_tags.csv',encoding='ISO-8859-1')

training_array=training_data.as_matrix()

def qst_classifier(qst):
#test_array=[['What is your name?', 'What'],
#            ['When is the show happening?','When'],
#            ['Is there a cab available for airport?', 'Affirmative']]
            
    classifier = NaiveBayesClassifier(training_array)
    qst_postag=nltk.pos_tag(nltk.word_tokenize(qst))

    frst_word=qst_postag[0][0]
    frst_tag=qst_postag[0][1]
    
    if (frst_word in ('Is','Was')) or (frst_word in ('Who','What') and frst_tag == 'WP') or (frst_word == 'When' and frst_tag == 'WRB'):
        return str(classifier.classify(qst))
    else:
        return 'Unknown'
        


#func_arg = {"-a": qst_classifier}

#if __name__ == "__main__":
#    func_arg[sys.argv[1]](sys.argv[2])
        

               

