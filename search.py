#!/usr/bin/python
import sys
import getopt
from nltk.stem.porter import *
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
    # query = process_query(query_file);
    ScoreCalculator = VectorSpaceModel(dictionary, postings_file, line_positions)
    scores = ScoreCalculator.getScores(query_file, last_line_pos)
    write_to_output_file(output_file, scores)

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