#!/usr/bin/python
import nltk
import sys
import getopt
import math
import string
from nltk.stem.porter import *
import xml.etree.ElementTree as et
from VectorSpaceModel import VectorSpaceModel

def search(query_file, dictionary_file, postings_file, output_file):
    """
    reads in and executes queries with the content of dictionary and postings file
    and writes the answers to the output_file
    
    Arguments:
        query_file:     path to the queries file, each query written on a new line
        dictionary_file path to the dictionary file 
                        (form: "term frequency line-in-postings-file")  
        postings_file:  path to postings file, each posting corresponding to different dictionary entries
                        (form: "file_name <space> file_name <space>...")
        output_file:    path to the file where the output should be written
        
    """
    
    dictionary = read_dict(dictionary_file)   
    line_positions, last_line_pos = get_line_positions(postings_file);
    query = process_query(query_file);
    ScoreCalculator = VectorSpaceModel(query, dictionary, postings_file, line_positions, last_line_pos)
    scores = ScoreCalculator.getScores()
    write_to_output_file(output_file, scores)

    print "======== NEW TEST RUN ========="
    print query
    print "\n"    

def read_dict(dictionary_file):
    """
    read dictionary_file and transfers it to the data structure of a python dictionary
    
    Arguments:
        dictionary_file   path to the file where the dictionary is saved
    
    Returns:
        dict structure
    """
    
    d = open(dictionary_file, 'r')

    dictionary = {}
    for line in d:
        word, freq, postings_line = line.split(' ')
        dictionary[word] = (freq, postings_line)

    d.close()
    return dictionary

# TODO: save this position in the dictionary instead of the line number
def get_line_positions(postings_file):
    """
    reads line positions in memory

    Returns:
        line_offset (positions) list that contains the starting position of each line in a file 
        last_line_pos   position for the last line (containing documents and their lengths)
    """
    
    file = open(postings_file, 'r')
    line_offset = []
    offset = 0
    for line in file:
        line_offset.append(offset)
        last_line_pos = offset
        offset += len(line)
    file.close()
    return (line_offset, last_line_pos)

def process_query(query_file):
    """
    read and process query by extracting title and description and calculating document scores

    Arguments:
        query_file      path to the file containing the query

    Returns: 
        query          a list with words extracted from the query file
    """

    tree = et.parse(query_file)
    root = tree.getroot()

    for child in root:
        if child.tag == 'title':
            title_content = child.text.encode('utf-8')

        if child.tag == 'description':
            desc_content = child.text.encode('utf-8')
            # remove the 4 first words since they always will be: "Relevant documents will describe"
            desc_content = ' '.join(desc_content.split()[4:])

    content = title_content + ' ' + desc_content

    stemmer = PorterStemmer()
    
    query_list = []
    sentences = nltk.sent_tokenize(content)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:
            # skip all words that contain just one item of punctuation
            if word in string.punctuation: 
                continue
            # add stemmed word the query_list
            query_list.append(word_stemming(word, stemmer))
    return query_list

def word_stemming(word, stemmer):
       """
           returns an uncapitalized and stemmed word 
       """
       return stemmer.stem(word.lower())

def write_to_output_file(output_file, scores):
    """
    writes the scores to output_file
    """
    f = open(output_file, 'w+')
    g = open('output-debug.txt', 'w+')
    for doc_name, score in scores:
        f.write(doc_name + ' ')
        g.write(doc_name + ' ' + str(score) + '\n')

def usage():
    print 'usage: ' + sys.argv[0] + ' -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results'


######################
# MAIN
######################

query_file = 'queries/q1.xml'
dictionary_file = 'dictionary.txt'
postings_file = 'postings.txt'
output_file = 'output.txt'

last_dict_line = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], 'q:d:p:o:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-q':
        query_file = a
    elif o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-o':
        output_file = a
    else:
        assert False, 'unhandled option'

search(query_file, dictionary_file, postings_file, output_file)