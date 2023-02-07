"""
ACKNOWLEDGMENTS:
* This code implements "The Toronto Paper Matching System: An automated paper-reviewer
assignment system," Charlin and Zemel, 2013  (which is not open source)
* Initial version was written by Han Zhao as part of the paper: "On strategyproof conference peer review," Xu, Zhao, Shi, and Shah, 2019. 
* Please acknowledge the aforementioned papers if you use this code.
* Code subsequently modified by Nihar Shah to enable for a more general purpose use

"""

import glob
import math
import os
from collections import Counter
from collections import defaultdict

import numpy as np
from nltk.stem import PorterStemmer

import re
import unicodedata


def isUIWord(word): #Returns true if the word is un-informative (either a stopword or a single character word)

    if len(word) <= 1:
        return True

    # List provided by Amit Gruber
    l = set(['a', 'about', 'above', 'accordingly', 'across', 'after', 'afterwards', 'again', \
             'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', \
             'always', 'am', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', \
             'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', \
             'around', 'as', 'aside', 'at', 'away', 'back', 'be', 'became', 'because', \
             'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', \
             'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', \
             'bottom', 'briefly', 'but', 'by', 'call', 'came', 'can', 'cannot', 'cant', \
             'certain', 'certainly', 'co', 'computer', 'con', 'could', 'couldnt', 'cry', \
             'de', 'describe', 'detail', 'do', 'does', 'done', 'down', 'due', 'during', \
             'each', 'edit', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', \
             'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', \
             'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', \
             'first', 'five', 'following', 'for', 'former', 'formerly', 'forty', 'found', \
             'four', 'from', 'front', 'full', 'further', 'gave', 'get', 'gets', 'give', \
             'given', 'giving', 'go', 'gone', 'got', 'had', 'hardly', 'has', 'hasnt', 'have', \
             'having', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', \
             'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', \
             'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', \
             'it', 'its', 'itself', 'just', 'keep', 'kept', 'kg', 'knowledge', 'largely', \
             'last', 'latter', 'latterly', 'least', 'less', 'like', 'ltd', 'made', 'mainly', \
             'make', 'many', 'may', 'me', 'meanwhile', 'mg', 'might', 'mill', 'mine', 'ml', \
             'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', \
             'name', 'namely', 'nearly', 'necessarily', 'neither', 'never', 'nevertheless', \
             'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'normally', 'not', \
             'noted', 'nothing', 'now', 'nowhere', 'obtain', 'obtained', 'of', 'off', \
             'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', \
             'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'owing', 'own', 'part', \
             'particularly', 'past', 'per', 'perhaps', 'please', 'poorly', 'possible', \
             'possibly', 'potentially', 'predominantly', 'present', 'previously', \
             'primarily', 'probably', 'prompt', 'promptly', 'put', 'quickly', 'quite', \
             'rather', 're', 'readily', 'really', 'recently', 'refs', 'regarding', \
             'regardless', 'relatively', 'respectively', 'resulted', 'resulting', 'results', 'rst', \
             'said', 'same', 'second', 'see', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'serious', \
             'several', 'shall', 'she', 'should', 'show', 'showed', 'shown', 'shows', 'side', \
             'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty', \
             'slightly', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', \
             'sometimes', 'somewhat', 'somewhere', 'soon', 'specifically', 'state', 'states', \
             'still', 'strongly', 'substantially', 'successfully', 'such', 'sufficiently', \
             'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'theirs', 'them', \
             'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', \
             'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', \
             'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', \
             'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', \
             'under', 'unless', 'until', 'up', 'upon', 'us', 'use', 'used', 'usefully', \
             'usefulness', 'using', 'usually', 'various', 'very', 'via', 'was', 'we', 'well', \
             'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', \
             'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', \
             'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'widely', \
             'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', \
             'yourself', 'yourselves'])

    if word in l:
        return True

    return False


def tokenize(line):
    space_regexp = re.compile('[^a-zA-Z]') 
    line = sanitize(line)  # sanitize returns unicode
    words = re.split(space_regexp, line)
    words = [x for x in words if len(x) > 0]

    return words


def sanitize(w): # remove accents and standardizes
    map = {'æ': 'ae',
           'ø': 'o',
           '¨': 'o',
           'ß': 'ss',
           'Ø': 'o',
           '\xef\xac\x80': 'ff',
           '\xef\xac\x81': 'fi',
           '\xef\xac\x82': 'fl'}

    # This replaces funny chars in map
    for char, replace_char in map.items():
        w = re.sub(char, replace_char, w)

    # This gets rite of accents
    w = ''.join((c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn'))

    return w


def paper2bow(text):
    """
    Tokenize and filter.
    :param text: string. Collection of texts from each paper.
    :return: map: string->int.
    """
    words = [w.lower() for w in tokenize(text)]
    # Filter out uninformative words.
    words = filter(lambda w: not isUIWord(w), words)
    # Use PortStemmer.
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    return Counter(words)


def compute_similarities(reviewer_folder, paper_folder):
    current_folder = os.getcwd()
    all_words = set()
    reviewers, papers = {}, {}
    # Parse all the reviewers.
    os.chdir(reviewer_folder)
    
    for textfile in glob.glob("*.txt"):
        with open(textfile, "r", encoding="utf-8") as fin:
            text = fin.read().lower()
            counter = paper2bow(text)
            reviewers[textfile] = counter
    
    os.chdir(current_folder)
    # Parse all the arxiv papers. 
    os.chdir(paper_folder)
    for textfile in glob.glob("*.txt"):
        with open(textfile, "r", encoding="utf-8") as fin:
            text = fin.read().lower()
            counter = paper2bow(text)
            papers[textfile] = counter
    
    # Combine all the unique words from both papers and reviewers.
    for reviewer in reviewers:
        all_words |= set(reviewers[reviewer].keys())
    for paper in papers:
        all_words |= set(papers[paper].keys())
    
    # Computing the idf of each unique word.
    N = len(reviewers) + len(papers)
    idf = defaultdict(lambda: 0.0)
    for reviewer in reviewers:
        for word in reviewers[reviewer]:
            idf[word] += 1.0
    for paper in papers:
        for word in papers[paper]:
            idf[word] += 1.0
    for word in idf:
        idf[word] = math.log(N / idf[word])
    assert len(all_words) == len(idf)
    # Use the order in idf.keys() as the default mapping from word to index. 
    #Start building similarity matrix
    os.chdir(current_folder)
    num_reviewers = len(reviewers)
    num_papers = len(papers)
    reviewer_idx = dict(zip(reviewers.keys(), list(range(num_reviewers))))
    paper_idx = dict(zip(papers.keys(), list(range(num_papers))))
    similarity_matrix = -np.ones((num_reviewers, num_papers))
    for reviewer in reviewers:
        for paper in papers:
            aid, pid = reviewer_idx[reviewer], paper_idx[paper]
            avec, pvec = reviewers[reviewer], papers[paper]
            if len(avec) == 0 or len(pvec) == 0:
                continue
            a_tot, p_tot = max(avec.values()), max(pvec.values())
            sim = 0.0
            # Compute the L2 norm of both avec and pvec.
            avec_norm, pvec_norm = 0.0, 0.0
            for word in avec:
                a_tf = 0.5 + 0.5 * avec[word] / a_tot
                w_idf = idf[word]
                avec_norm += (a_tf * w_idf) ** 2
                if word in pvec:
                    # Augmented term frequency to prevent a bias towards longer document.
                    p_tf = 0.5 + 0.5 * pvec[word] / p_tot
                    sim += a_tf * p_tf * (w_idf ** 2)
            for word in pvec:
                p_tf = 0.5 + 0.5 * pvec[word] / p_tot
                w_idf = idf[word]
                pvec_norm += (p_tf * w_idf) ** 2
            # Compute the cosine angle as the similarity score.
            avec_norm, pvec_norm = math.sqrt(avec_norm), math.sqrt(pvec_norm)
            similarity_matrix[aid, pid] = sim / avec_norm / pvec_norm
    
    print(reviewer_idx, paper_idx, similarity_matrix)

