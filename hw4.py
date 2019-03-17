# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 00:09:01 2019

@author: mwon579
"""

# import libraries
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import string
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


'''
Pre Processing
'''
# feature: checck if actual numbers exist in percent token 
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    isnumber = is_number(word)
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
        'number': isnumber,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }
    
class ConsecutiveNERChunkTagger(nltk.TaggerI): 
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(
            train_set)
    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNERChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNERChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

import pickle
f = open('my_classifier.pickle', 'rb')
full_model = pickle.load(f)
f.close()

# Hardcoded word lists
stop_words=sorted(set(stopwords.words("english")))

corpus2013 = []

for file in os.listdir('2013'):
    corpus2013.append(open(os.path.join('2013',file),'rb').read().decode('ANSI'))

word_tokens_corpus = [nltk.word_tokenize(document.lower()) for document in corpus2013]
sent_tokens_corpus = [nltk.sent_tokenize(document) for document in corpus2013]
pos_taggedCorpus = [[nltk.pos_tag(nltk.word_tokenize(sent)) for sent in doc] for doc in sent_tokens_corpus]
NER_chunkedCorpus = [[full_model.parse(pos_sent) for pos_sent in doc] for doc in pos_taggedCorpus]

# NE_chunkedCorpus = [[nltk.ne_chunk(pos_sent) for pos_sent in doc] for doc in pos_taggedCorpus]


stemmer = SnowballStemmer('english')

from nltk.corpus import wordnet as wn

#for ss in wn.synsets('firm'):
#    for hyper in ss.hypernyms():
#        print(hyper.name().split('.')[0])

company_hyponyms = []
for ss in wn.synsets('company'):
    for hyper in ss.hypernyms():
        company_hyponyms.append(hyper.name().split('.')[0])

company_hyponyms = ['institution', 'organization', 'business']

ceo_hyponyms = []
for ss in wn.synsets('CEO'):
    for hyper in ss.hypernyms():
        ceo_hyponyms.append(hyper.name().split('.')[0])

ceo_hyponyms = ['corporate_executive']

'''
Stem all the words in the corpus
'''
for doc in word_tokens_corpus:
    for i in range(len(doc)):
        doc[i] = stemmer.stem(doc[i])
        
def identity(x):
    return x

# TF
tf = CountVectorizer(tokenizer=identity,preprocessor=identity,
                             ngram_range=(1, 1),stop_words=stop_words)

doc_term_mat_tf = tf.fit_transform(word_tokens_corpus)

#a = pd.DataFrame(doc_term_mat_tf,index=np.linspace(0,364,365), columns=tf.get_feature_names())
#
#b =pd.DataFrame(doc_term_mat_tf)

# TFiDF
tfidf = TfidfVectorizer(tokenizer=identity,preprocessor=identity,
                             ngram_range=(1, 1),stop_words=stop_words)

doc_term_mat_tfidf = tfidf.fit_transform(word_tokens_corpus)

docs_feature_names = tf.get_feature_names()








'''
Question Classifier
'''
yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

# Take in a tokenized question and return the question type and body
def processquestion(qwords):
    
    # Find "question word" (what, who, where, etc.)
    questionword = ""
    qidx = -1

    for (idx, word) in enumerate(qwords):
        if word.lower() in questionwords:
            questionword = word.lower()
            qidx = idx
            break
        elif word.lower() in yesnowords:
            return ("YESNO", qwords)

    if qidx < 0:
        return ("MISC", qwords)

    if qidx > len(qwords) - 3:
        target = qwords[:qidx]
    else:
        target = qwords[qidx+1:]
    typ = "MISC"

    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        for word in target:
            if word in ['CEO', 'ceo', 'corporate', 'executive' , 'chief', 'officer']:
                typ = "CEO"
#    elif questionword == "where":
#        typ = "PLACE"
#    elif questionword == "when":
#        typ = "TIME"
#    elif questionword == "how":
#        if target[0] in ["few", "little", "much", "many"]:
#            typ = "QUANTITY"
#            target = target[1:]
#        elif target[0] in ["young", "old", "long"]:
#            typ = "TIME"
#            target = target[1:]

    # Trim possible extra helper verb
    elif questionword in ["which", 'what']:
#        target = target[1:]
        for word in target:
            if word in ['company', 'companies', 'business', 'business', 'firm', 'firms']:
                typ = "COMP"
    elif questionword == 'what':
        for word in target:
            if word in ['%','percent', 'percentage', 'percentile']:
                typ = "PCT"
    elif target[0] in yesnowords:
        target = target[1:]
    
    # Return question data
    return (typ, target)







'''
Query
'''
def QA():
    query = input("What is your question? ")
    
    qwords = nltk.word_tokenize(query)
    
    if qwords[-1] in ['?','.', '!']:
        qwords = qwords[:-1]
    
    for i in range(len(qwords)):
        qwords[i] = qwords[i].lower()
    
    typ, key_words = processquestion(qwords)
    stem_kwords = []
    for word in key_words:
        stem_kwords.append(stemmer.stem(word))
    
    hm = {}
    
    words_to_rmv = []
    for i in range(len(stem_kwords)):
        if stem_kwords[i] in docs_feature_names:
            if typ == 'PCT' and stem_kwords[i] in ['%', 'percent']:
                words_to_rmv.append(stem_kwords[i])
            elif stem_kwords[i] in ["'s", 'compani', 'year', 'month']:
                words_to_rmv.append(stem_kwords[i])
            else:
                hm[stem_kwords[i]] = []
        else:
            words_to_rmv.append(stem_kwords[i])
    
    for word in words_to_rmv:
        stem_kwords.remove(word)
    
    #if typ == 'PCT':
    #    stem_kwords += ['%', 'percent']
        
        
    #for i in range(len(word_tokens_corpus)):
    #    for word in stem_kwords:
    #        w = docs_feature_names.index(word)
    #        if doc_term_mat_tf[i, w] > 0:
    #            hm[word].append(i)
    #            break        
    
    candidates = []
    
    for i in range(len(word_tokens_corpus)):
        include = 1
        for word in stem_kwords:
            w = docs_feature_names.index(word)
            if doc_term_mat_tf[i, w] == 0:
                include = 0
                break
        if include:
            candidates.append(i)
    
    for key in hm.keys():
        candidates += hm[key]
    
    scores = []
    for candidate in candidates:
        score = 0
        for word in stem_kwords:
            if word in docs_feature_names:
                w = docs_feature_names.index(word)
                ws = doc_term_mat_tfidf[candidate, w]
                score += ws
        scores.append(score)
    
    df = pd.DataFrame(np.mat([candidates, scores]).T,columns = ['candidate','score'])
    
    top_candidates = list(df.sort_values('score', ascending=False)['candidate'].astype(int))[0:50]
    
    sentences = []
    #pos_taggedSentences = []
    #word_tokens_sents = []
    NER_chunkedSentences = []
    for i in top_candidates:
        sents = sent_tokens_corpus[i]
    #    pos_sents = pos_taggedCorpus[i]
    #    word_tokens_sents.append(word_tokens_corpus[i])
        NER_sents = NER_chunkedCorpus[i]
        for j in range(len(sents)):
            sentences.append(sents[j])
    #        pos_taggedSentences.append(pos_sents[j])
            NER_chunkedSentences.append(NER_sents[j])
            #pos_taggedSent = nltk.pos_tag(nltk.word_tokenize(sent))
            #pos_taggedSentences.append(pos_taggedSent)
    
    #sentences = []
    #NER_chunkedSentences = []
    #for i in top_candidates:
    #    sents = sent_tokens_corpus[i]
    #    for sent in sents:
    #        sentences.append(sent)
    #        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
    #        NER_chunkedSentences.append(full_model.parse(pos_tagged))
           
    
    word_tokens_sents = [nltk.word_tokenize(sent.lower()) for sent in sentences]
    
    for sent in word_tokens_sents:
        for i in range(len(sent)):
            sent[i] = stemmer.stem(sent[i])
    
    sents_term_tfidf = tfidf.fit_transform(word_tokens_sents)
    
    sents_feature_names = tfidf.get_feature_names()
    
    corpus_len = len(sentences)
    
    candidate_sents = []
    
    if typ == 'CEO':
        for i in range(corpus_len):
            for index, subtree in enumerate(NER_chunkedSentences[i]):
                if (type(subtree) != tuple) and (subtree.label() == 'CEO'):
                    candidate_sents.append(i)
                    break
    
    elif typ == 'COMP':
        for i in range(corpus_len):
            for index, subtree in enumerate(NER_chunkedSentences[i]):
                if (type(subtree) != tuple) and (subtree.label() in ['COMP','CEO']):
                    candidate_sents.append(i)
                    break
    
    elif typ == 'PCT':
        for i in range(corpus_len):
            for index, subtree in enumerate(NER_chunkedSentences[i]):
                if (type(subtree) != tuple) and (subtree.label() == 'PCT'):
                    candidate_sents.append(i)
                    break        
    
    elif typ == 'MISC':
        candidate_sents = range(len(sentences))
    #    for i in range(corpus_len):
    #    include = 1
    #    for word in stem_kwords:
    #        w = sents_feature_names.index(word)
    #        if sents_term_tfidf[i, w] == 0:
    #            include = 0
    #            break
    #    if include:
    #        candidate_sents.append(i)
    
    score_sents = []
    for candidate in candidate_sents:
        score = 0
        for word in stem_kwords:
            if word in sents_feature_names:
                w = sents_feature_names.index(word)
                ws = sents_term_tfidf[candidate, w]
                if typ == 'PCT' and word == stem_kwords[-1]:
                    ws = 3*ws # add more weight to the last word (unemployment, inflation, etc...)
                score += ws
    #                if ws > 0:
    #                    if word == 'bankrupt':
    #                        multiplier += 2
    #                    else:   
    #                        multiplier += 1
        score_sents.append(score)
    
    df = pd.DataFrame(np.mat([candidate_sents, score_sents]).T,columns = ['candidate','score'])
    top_candidates = list(df.sort_values('score', ascending=False)['candidate'].astype(int))[0:10]
    
    #for i in top_candidates:    
    #    print(sentences[i])
    
    best = sentences[top_candidates[0]]
        
    # helper function to untag chunks and extract the string only
    def extract_str(subtree):
        return ' '.join(word for word, tag in subtree.leaves())
    
    percent_tokenizer = RegexpTokenizer("-?\w+(?:\.\w+)? ?%|-?\w+(?:\.\w+)? percent(?:age points?|ile (?:points?)?)?")
    if typ == 'CEO':
        tagged_sent = full_model.parse(nltk.pos_tag(nltk.word_tokenize(best)))
        for index, subtree in enumerate(tagged_sent):
                if (type(subtree) != tuple) and (subtree.label() == 'CEO'):
                    s = extract_str(subtree)
                    print(s)
    elif typ == 'COMP':
        tagged_sent = full_model.parse(nltk.pos_tag(nltk.word_tokenize(best)))
        for index, subtree in enumerate(tagged_sent):
                if (type(subtree) != tuple) and (subtree.label() == 'COMP'):
                    s = extract_str(subtree)
                    print(s)
    elif typ == 'PCT':
    #    for i in top_candidates:    
    #        print(sentences[i])
        print(percent_tokenizer.tokenize(best)[0]) 
    
    elif typ == 'MISC':
        for i in top_candidates:    
            print(sentences[i])

# Who is Apple's current CEO? Tim Cook
# Who is the CEO of Microsoft? Steve Ballmer
# Who is the Amazon CEO? Jeff Bezos
# Who's the CEO of IBM? Ginni Rommetty

# Which companies went bankrupt in September 2008? F/F
# Which companies went bankrupt in 2009?
# Which companies went bankrupt in October?
    
# what affects GDP? Exchange rate, inflaction rate, 