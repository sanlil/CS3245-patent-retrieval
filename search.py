#!/usr/bin/python
import sys
import getopt
from nltk.stem.porter import *
from VectorSpaceModel import VectorSpaceModel

def search(query_file, dictionary_file, postings_file, output_file, patent_info_file):
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
    patent_info = get_patent_info(patent_info_file)
    ScoreCalculator = VectorSpaceModel(dictionary, postings_file, line_positions)
    scores = ScoreCalculator.getScores(query_file, last_line_pos)
    write_to_output_file(output_file, scores)

    print "======= NEW TEST RUN ========="
    count = 0
    while count < 20:
        doc_name, score = scores[count]
        print doc_name, score, patent_info[doc_name]
        count += 1
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
    
    f = open(postings_file, 'r')
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        last_line_pos = offset
        offset += len(line)
    f.close()
    return (line_offset, last_line_pos)

def write_to_output_file(output_file, scores):
    """
    writes the scores to output_file
    """
    f = open(output_file, 'w+')
    g = open("debug_" + output_file, 'w+') #just for debugging - remove later
    
	# Fred: output has no trailing space, and ends with a newline
    f.write(' '.join([doc_name for doc_name, score in scores]) + '\n')
    
    g.write('\n'.join([doc_name + ' ' + str(score) for doc_name, score in scores])) #just for debugging - remove later	

def get_patent_info(patent_info_file):
    """
        Returns:
            patent_info     dictionary with all patent meta data accessible by patent name
                            meta data format: [year, cites, ipc, inventor]
    """
    patent_info = {}
    f = open(patent_info_file, 'r')
    for line in f:
        info_list = line.split(" | ")
        doc_name = info_list[0]
        patent_info[doc_name] = info_list[1:]
    return patent_info

def usage():
    print 'usage: ' + sys.argv[0] + ' -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results'


######################
# MAIN
######################

query_file = 'queries/q1.xml'
dictionary_file = 'dictionary.txt'
postings_file = 'postings.txt'
output_file = 'output.txt'
patent_info_file = 'patent_info.txt'

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

search(query_file, dictionary_file, postings_file, output_file, patent_info_file)