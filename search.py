#!/usr/bin/python
import nltk
import sys
import getopt
import math
from nltk.stem.porter import *
import xml.etree.ElementTree as et

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
    line_positions = get_line_positions(postings_file);

    output_table = []
    f = open(query_file, 'r')

    for line in f:
        query = tokenize_query(line)
        
        #TODO: Calculate scores for all documents

        output_table.append(result_list)

    output_table = process_query(query_file);
    write_to_output_file(output_file, output_table)
    

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
        word, frequency, postings_line = line.split(' ')
        dictionary[word] = (freq, postings_line)

    d.close()
    return dictionary


def get_line_positions(postings_file):
    """
    reads line positions in memory

    Returns:
        line_offset (positions) list that contains the starting position of each line in a file 
    """
    
    file = open(postings_file, 'r')
    line_offset = []
    offset = 0
    for line in file:
        line_offset.append(offset)
        offset += len(line)
    file.close()
    return line_offset


def process_query(query_file):
    """
    read and process query line by line by calculating document scores

    Arguments:
        query_file      path to the file containing the queries

    Returns: 
        output_table    a matrix with the result for each query on corresponding line
    """

    output_table = []
    f = open(query_file, 'r')
    for line in f:
        query = tokenize_query(line)
        
        #TODO: calculate scores

        output_table.append(result)
    f.close()
    return output_table


def tokenize_query(line):
    return nltk.word_tokenize(line)


def word_stemming(word, stemmer):
    """
        returns an uncapitalized and stemmed word 
    """
    return stemmer.stem(word.lower())


def write_to_output_file(output_file, output_table):
    """
    writes output_table to output_file
    """
    f = open(output_file, 'w+')
    for line in output_table:
        for doc_nr in line:
            f.write(str(doc_nr) + " ")
        f.write("\n")


def usage():
    print "usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results"


######################
# MAIN
######################

query_file = "queries.txt"
dictionary_file = "dictionary.txt"
postings_file = "postings.txt"
output_file = "output.txt"

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
        assert False, "unhandled option"

search(query_file, dictionary_file, postings_file, output_file)