#!/usr/bin/python
import nltk
import sys
import getopt
import os
import string
import math
import re
from nltk.stem.porter import *
import xml.etree.ElementTree as et
import text_processing

def indexing(training_path, postings_file, dictionary_file, patent_info_file):
    """
    Create an index of the corpus at training_path, placing the index in
    dictionary_file, postings_file and patent_info_file.
    
    dictionary_file will contain a list of the terms contained in the corpus.
    On every line, the following information will be included (separated by spaces):
        <term indexed> <document frequency> <postings pointer>
    The postings pointer is a line number in the postings file.
    Every line of the postings file is a postings list for the term that points to that
    line. A line contains several entries. Each entry has the following form:
        <patent ID> <log tf> <tf> <list of positions>
    The list of positions indicates the word positions at which the indexed word can
    be found in the given patent (a standard positional index). The list is always
    <tf> elements long.
    """
    pats = sorted(os.listdir(training_path))
    
    pi = open(patent_info_file, 'w')
    postings = dict()
    for pat in pats:
        tree = et.parse(os.path.join(training_path, pat))
        
        pat_id = os.path.splitext(pat)[0]
        
        root = tree.getroot()
        title_content = ""
        abstract_content = ""

        year = ""
        cites = "0"
        ipc = ""
        inventor = ""
        content = ""
        for child in root:
            # extract patent content
            if child.get('name') == 'Title':
                content += child.text.encode('utf-8') + " "

            if child.get('name') == 'Abstract':
                content += child.text.encode('utf-8') + " "

            # extract patent info (meta data)
            if child.get('name') == 'Publication Year':
                year = child.text.encode('utf-8').strip()

            if child.get('name') == 'Cited By Count':
                cites = child.text.encode('utf-8').strip()

            if child.get('name') == 'IPC Primary':
                ipc = child.text.encode('utf-8').strip()

            if child.get('name') == '1st Inventor':
                inventor = child.text.encode('utf-8').strip()

        pi.write(pat_id + " | " + year + " | " + cites +  " | " + ipc + " | " + inventor + "\n")
        
        stemmer = PorterStemmer()
        # remove non utf-8 characters, http://stackoverflow.com/a/20078869
        content = re.sub(r'[^\x00-\x7F]+',' ', content)

        sentences = nltk.sent_tokenize(content)
        i = 0
        occurences = dict()
        for sentence in sentences:
            # tokenize sentences in words
            words = nltk.word_tokenize(sentence)
            for word in words:
                normalized = text_processing.normalize(word)
                if normalized is None:
                    continue
                
                if normalized in occurences:
                    occurences[normalized].append(i)
                else:
                    occurences[normalized] = [i]
                
                i += 1
        
        for word, positions in occurences.iteritems():
            if word in postings:
                postings[word].append((pat_id, positions))
            else:
                postings[word] = [(pat_id, positions)]
        
    pi.close()
    write_dict_postings(postings, postings_file, dictionary_file, training_path)    
    
def write_dict_postings(data, postings_file, dictionary_file, training_path):
    f = open(postings_file, 'w')
    d = open(dictionary_file, 'w')

    line_index = 0
    current_word = ""
    current_word_freq = {}
    doc_lengths = {}
    for word, postings in data.iteritems():
        doc_freq = len(postings)
        d.write(word + ' ' + str(doc_freq) + ' ' + str(line_index) + '\n')
        for pat, positions in postings:
            tf = len(positions)
            log_tf = 0 if tf == 0 else 1 + math.log(tf)
            doc_lengths[pat] = doc_lengths.get(pat, 0.0) + log_tf * log_tf
            
            f.write(pat + ' ' + str(log_tf) + ' ' + str(len(positions)) + ' ')
            f.write(' '.join([str(p) for p in positions]) + ' ')
        f.write('\n')    
        line_index += 1

    for doc, length in doc_lengths.iteritems():
        f.write(doc + ' ' + str(math.sqrt(length)) + ' ')

    d.write("# " + training_path)

    f.close()
    d.close()

def usage():
    print "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file -q patent-info-file"


######################
# MAIN
######################

training_path = "patsnap-corpus/"
dictionary_file = "dictionary.txt"
postings_file = "postings.txt"
patent_info_file = "patent_info.txt"

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:q:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        training_path = a
        if "/" not in training_path:
            training_path += "/"
    elif o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        patent_info_file = a
    else:
        assert False, "unhandled option"

indexing(training_path, postings_file, dictionary_file, patent_info_file)
