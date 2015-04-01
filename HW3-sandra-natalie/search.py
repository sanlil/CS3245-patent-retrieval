#!/usr/bin/python
import nltk
import sys
import getopt
import math
# import time
import numpy as np
from nltk.stem.porter import *

def search(query_file, dictionary_file, postings_file, output_file):
    """
    reads in and executes queries with the content of dictionary and postings file
    and writes the answers to the output_file
    
    Arguments:
        query_file:     contains different queries, each written in a new line
        dictionary_file:contains a dictionary that is written down in a file 
                        (form: "term frequency line-in-postings-file")  
        postings_file:  contains all postings belonging to the different dictionary entries
                        (form: "file_name <space> file_name <space>...")
        output_file:    shows all file numbers that result from the queries
                        (form: "file_name <space> file_name <space>...")
        
    """
    
    # get dict data structure - dictionary in memory
    d = open(dictionary_file, 'r')
    dictionary = read_dict(d)
    d.close()

    # index for lines in the postings file used for seek()
    line_positions = get_line_positions(postings_file);
    # number of documents
    all_postings = get_all_postings(dictionary, postings_file, line_positions)
    n = len(all_postings)

    #get a list with the normalization factor for all documents
    length_vector = get_document_lengths(postings_file, line_positions);
    
    output_table = []

    f = open(query_file, 'r')

    # read and process query line by line
    for line in f:
        # tokenize query
        query = tokenize_query(line)
        # calculate cosine scores per query
        scores = calculate_cosine_score(query, dictionary, postings_file, line_positions, all_postings, n, length_vector)
        # select 10 best results based on calculated scores
        top_10_docs = get_top_k_docs(scores, 10)

        output_table.append(top_10_docs)

    write_to_output_file(output_file, output_table)
    f.close()


def calculate_cosine_score(query, dictionary, postings_file, line_positions, all_postings, n, length_vector):
    """
    computes the cosine scores for a query and all given documents and returns the best results

    Arguments:
        query           list of query terms
        dictionary      
        postings_file
        line_positions  contains the position of every line in the postings-file
                        are used to read a specific line out of the postings file
        all_postings    array with all different file names that occur in training data set
        n               number of documents in training data
        length_vector   array that contains the normalization factor for each document accessable by document number


    Returns:
        cosine scores accessable per document number

    """
    
    # list of n scores initialized with 0 - length: highest document number
    scores = [0 for x in range(int(all_postings[n-1])+1)]

    # initializing the vectors for document weights - length: highest document number
    weight_documents = [[0 for x in range(len(query))] for y in range(int(all_postings[n-1])+1)]
    # initializing the vectors for query weights - length: query
    weight_query = [0 for x in range(len(query))]

    length_query = 0
    stemmer = PorterStemmer()

    for termNo in range(len(weight_query)):
        query[termNo] = word_stemming(query[termNo], stemmer)
        term = query[termNo]
        term_postings = get_postings(term, dictionary, postings_file, line_positions)
        weight_query[termNo] = get_weight_query_term(term, dictionary, n, query)
        length_query += math.pow(weight_query[termNo], 2)

        for (docNo, tf) in term_postings:
            weight_documents[docNo][termNo] = get_weight_doc_term(tf)
            scores[docNo] += weight_documents[docNo][termNo] * weight_query[termNo]
    length_query = math.sqrt(length_query)

    for docNo in all_postings:
        docNo = int(docNo)
        if (length_vector[docNo]*length_query) != 0:
            scores[docNo] = scores[docNo]/(length_vector[docNo]*length_query)
    return scores

def calculate_length(query, weight_documents, weight_query, docNo):
    """
    calculates the Euclidean normalization values (length) for the given document number
    cosine per document can be normalized by deviding through these values

    Arguments:
        query               array of all query terms
        weight_documents    double array of all weights per document and term [docNo][termNo]
        weight_query        array of all weights per term in a query [termNo]
        docNo               

    Returns:
        length              length value for given docNo (length of document vectors multiplied with query vector length) 
    """

    query_len = 0
    doc_len = 0
    length = 0
    # calculate the Euclidean normalization values for all documents
    for termNo in range(len(weight_query)):
        query_len += weight_query[termNo]**2
        doc_len += weight_documents[docNo][termNo]**2
    length = math.sqrt(query_len)*math.sqrt(doc_len)
    return length

def get_weight_query_term(term, dictionary, n, query):
    """
    calculates the weight of a term in a query by pattern tf.idf
    if term is not defined in dictionary => weight_query = 0

    Arguments:
        term        term in query to calculate for
        dictionary
        n           number of documents in training data
        query       array with query terms
    
    Returns:
        weight of query
    """
    # calculate tf in query
    tf = query.count(term)
    log_tf = 1 + math.log10(tf)

    if term in dictionary.keys():
        (freq, postings_line) = dictionary[term]
        # calculate idf
        idf = math.log10(float(n)/float(freq))
        return log_tf * idf
    else:
        return 0

def get_weight_doc_term(tf):
    """
    calculates the weight of a term in a document
    (as it is already calculated during indexing and saved in the postings_file just a float version is returned for further calculation)
    """
    return float(tf)

def get_top_k_docs(scores, k):
    """
    selects the best k documents based on the given scores per document
    (the higher to score the better)

    Arguments:
        scores      array with cosine scores accessable by document number
        k           number of top documents to be returned

    Returns:
        array with top k document numbers regarding query terms 
        (just documents with scores > 0 are returned => array might be smaller than k)

    Source:
        http://stackoverflow.com/a/6910672
    """
    if len(scores) < k:
        k = len(scores)
    # http://stackoverflow.com/a/6910672
    np_array = np.array(scores)
    top_k = np_array.argsort()[-k:][::-1]

    count = k - 1
    while count >= 0 and scores[top_k[count]] == 0:
        top_k = top_k[:-1]
        count -= 1
    return list(top_k)


def read_dict(d):
    """
    read dictionary_file and transfers it to the data structure of a python dictionary
    
    Arguments:
        d   dictionary file of form: term frequency line-in-postings-file
    
    Returns:
        dict structure
    """
    
    dictionary = {}
    count = 0;
    for line in d:
        line = line.split(' ')
        word = line[0].rstrip()
        freq = line[1].rstrip()
        postings_line = line[2].rstrip()
        dictionary[word] = (freq, postings_line)
        count += 1
    global last_dict_line
    last_dict_line = count
    return dictionary

def get_line_positions(postings_file):
    """
    reads line positions in memory

    Returns:
        line_offset (positions) list that contains the starting position of each line in a file 
    
    source:
        http://stackoverflow.com/questions/620367/python-how-to-jump-to-a-particular-line-in-a-huge-text-file
    """
    
    file = open(postings_file, 'r')
    line_offset = []
    offset = 0
    for line in file:
        line_offset.append(offset)
        offset += len(line)
    file.close()
    return line_offset

def tokenize_query(line):
    return nltk.word_tokenize(line)

def word_stemming(old_word, stemmer):
    """
        returns an uncapitalized and stemmed word 
    """
    return stemmer.stem(old_word.lower())

def get_postings(word, dictionary, postings_file, line_positions):
    """
    looks up a word in the given dictionary 
    and returns all postings that belong to that word
    
    Arguments:
        word            term that is meant to be look up
        dictionary  
        postings_file
        line_positions  contains the position of every line in the postings-file
                        are used to read a specific line out of the postings file
    
    Returns:    
        list of tuples (docNo, tf) belonging to word
    """

    # dictionary contains word
    if word in dictionary.keys():
        (freq, postings_line) = dictionary[word]
        f = open(postings_file, 'r')
        f.seek(line_positions[int(postings_line)])
        postings_list = f.readline().split()
        docNo_tf_array = []
        count = 0
        while count < len(postings_list):
            docNo_tf_array.append((int(postings_list[count]), float(postings_list[count+1])))
            count += 2
        f.close()
        return docNo_tf_array

    # dictionary does not contain word
    else:
        return []

def get_all_postings(dictionary, postings_file, line_positions):
    """
    Arguments:
        dictionary
        postings_file
        line_positions  contains the position of every line in the postings-file
                        are used to read a specific line out of the postings file
    Returns:
        array with all different file names that occur in training data set
    """
    
    f = open(postings_file, 'r')
    f.seek(line_positions[last_dict_line])
    result = f.readline().split()
    f.close()
    return result

def get_document_lengths(postings_file, line_positions):
    """
    Returns:
        the normalization factor for each document
    """
    
    f = open(postings_file, 'r')
    f.seek(line_positions[last_dict_line+1])
    length_vector = [float(x) for x in f.readline().split()]
    f.close()
    return length_vector

def write_to_output_file(output_file, output_table):
    """
    writs output_table to output_file
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