# -*- coding: utf-8 -*-
"""
Created on 2023-07-06 (Thu) 18:39:20

Input preprocessing

@author: I.Azuma
"""
#%%
import gensim
from urllib import request 
#%%
def _build_bow_vocab(data, bow_vocab_size, stopwords=None):

    bow_dictionary = gensim.corpora.Dictionary(data)

    if stopwords is None:
        res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt")
        stopwords = [line.decode("utf-8").strip() for line in res]
        res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/English.txt")
        stopwords += [line.decode("utf-8").strip() for line in res]

        stopwords += ['*', '&', '[', ']', ')', '(', '-',':','.','/','0', '...?', '——', '!【', '"', ')、', ')。', ')」']
        print("# Stopwords : ", len(stopwords))
    
    # Remove STOPWORDS
    STOPWORDS = gensim.parsing.preprocessing.STOPWORDS
    if stopwords is not None:
        STOPWORDS = set(STOPWORDS).union(set(stopwords))
    bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, STOPWORDS)))
    
    # Re-id
    bow_dictionary.filter_extremes(no_below=5, no_above=0.2, keep_n=bow_vocab_size)
    bow_dictionary.compactify()
    bow_dictionary.id2token = dict([(id, t) for t, id in bow_dictionary.token2id.items()])
    
    print("BOW dict length : %d" % len(bow_dictionary))
    
    return bow_dictionary

def build_bow_vocab(data, stopwords=None):

    bow_dictionary = gensim.corpora.Dictionary(data)

    if stopwords is None:
        res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt")
        stopwords = [line.decode("utf-8").strip() for line in res]
        res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/English.txt")
        stopwords += [line.decode("utf-8").strip() for line in res]

        stopwords += ['*', '&', '[', ']', ')', '(', '-',':','.','/','0', '...?', '——', '!【', '"', ')、', ')。', ')」']
        print("# Stopwords : ", len(stopwords))
    
    
    # Remove STOPWORDS
    STOPWORDS = gensim.parsing.preprocessing.STOPWORDS
    if stopwords is not None:
        STOPWORDS = set(STOPWORDS).union(set(stopwords))
    bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, STOPWORDS)))
    
    # Re-id
    bow_dictionary.filter_extremes(no_below=5, no_above=0.2)
    bow_dictionary.compactify()
    bow_dictionary.id2token = dict([(id, t) for t, id in bow_dictionary.token2id.items()])
    
    print("BOW dict length : %d" % len(bow_dictionary))
    
    return bow_dictionary