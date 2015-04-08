#!/usr/bin/python
import sys
import getopt
from nltk.stem.porter import *
from VectorSpaceModel import VectorSpaceModel

DEBUG_RESULTS = True

def print_result_info(scores, retrieve, not_retrieve):
    """
    Print some useful information on the query results based on the expected results.
    Tell us which relevant documents we missed and what are the rankings of the relevant
    documents we retrieved. Also tell us which irrelevant documents were retrieved,
    and what are their rankings.
    """

    # Create a dictionary of the positions of the documents
    positional_scores = [(doc, i + 1) for i, (doc, score) in enumerate(scores)]
    scores_dict = dict(positional_scores)
    retrieved_docs = set(scores_dict.keys())
    
    # Obtain the documents that we should retrieve and those that we shouldn't
    with open(retrieve) as r, open(not_retrieve) as n:
        wanted = set([l.rstrip() for l in r.readlines()])
        unwanted = set([l.rstrip() for l in n.readlines()])
        
    print 'Retrieved %d documents' % len(scores)
    
    correctly_obtained = wanted & retrieved_docs
    nco = len(correctly_obtained)
    nw = len(wanted)
    print 'Correctly hit %d documents out of %d wanted (%f%%)' % (nco, nw, nco / float(nw) * 100.0)
    print 'Documents missed: %s\n' % ', '.join(wanted - correctly_obtained)
    
    print 'Rankings of hit documents:\n\t',
    print '\n\t'.join([doc + ':' + ((20 - len(doc)) * ' ') + str(rank) for doc, rank in positional_scores if doc in correctly_obtained])
    
    print ''
    
    incorrectly_obtained = unwanted & retrieved_docs
    nio = len(incorrectly_obtained)
    nu = len(unwanted)
    print 'Incorrectly retrieved %d documents out of %d to avoid (%f%%)' % (nio, nu, nio / float(nu) * 100.0)
    print 'Rankings of false positives:\n\t',
    print '\n\t'.join([doc + ':' + ((20 - len(doc)) * ' ') + str(rank) for doc, rank in positional_scores if doc in incorrectly_obtained])
    
    

def search(query_file, dictionary_file, postings_file, output_file, patent_info_file, retrieve, not_retrieve):
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
    
    if DEBUG_RESULTS:
        print_result_info(scores, retrieve, not_retrieve)

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
    
    f = open(postings_file, 'rU')
    line_offset = []
    offset = 0
    for line in f:
        line_offset.append(offset)
        last_line_pos = offset
        offset += len(line)
        
        # Make sure we support windows-style endings correctly
        if f.newlines == '\r\n':
            offset += 1
    f.close()
    return (line_offset, last_line_pos)

def write_to_output_file(output_file, scores):
    """
    writes the scores to output_file
    """
    f = open(output_file, 'w+')
    
    #just for debugging - remove later
    output_path_parts = output_file.split('/')
    output_path_parts[len(output_path_parts)-1] = 'debug_' + output_path_parts[len(output_path_parts)-1]
    print "output_path_parts",output_path_parts
    debug_output_path = '/'.join(output_path_parts)
    print "debug_output_path",debug_output_path
    g = open(debug_output_path, 'w+')
    
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
    print 'usage: ' + sys.argv[0] + ' -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results -r output-debug-file'


######################
# MAIN
######################

query_file = 'queries/q2.xml'
dictionary_file = 'dictionary.txt'
postings_file = 'postings.txt'
output_file = 'output.txt'
patent_info_file = 'patent_info.txt'
retrieve = 'queries/q2-qrels+ve.txt'
not_retrieve = 'queries/q2-qrels-ve.txt'

last_dict_line = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], 'q:d:p:o:r:n:')
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
    elif o == '-r':
        retrieve = a
    elif o == '-n':
        not_retrieve = a
    else:
        assert False, 'unhandled option'

search(query_file, dictionary_file, postings_file, output_file, patent_info_file, retrieve, not_retrieve)
