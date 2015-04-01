#!/usr/bin/python
import nltk
import sys
import getopt
import os
import string
import math
from nltk.stem.porter import *


def indexing(training_path, postings_file, dictionary_file):
    """
    reads the training data and creates a postings and a dictionary file
    
    Arguments:
        training_path       folder containing all training data files to be used
        dictionary_file:    contains a dictionary in the end that is written down in a file 
                            (form: "term frequency line-in-postings-file(-->'pointer')")    
        postings_file:      contains all postings belonging to the different dictionary entries
                            form: "file_name <space> file_name <space>..."
                            the pre-last line contains the names of all given documents in the training data
                            the last line contains the length vector (all normalization factors from 0 until the highest document ID)
    """
    
    (word_list, file_list) = read_training_data(training_path)
    sort_word_list = sort_alphabetically(word_list)
    file_list = sort_numerically(file_list)

    consolidate_list(sort_word_list, file_list, postings_file, dictionary_file)


def sort_alphabetically(word_list):
    return sorted(word_list,key=lambda x: x[0])


def sort_numerically(file_list):
    file_list = [int(x) for x in file_list]
    return file_list.sort()


def read_training_data(training_path):
    """
    reads all files that can be found in the given folder of training data,
	tokenizes the content, stems the words 
	and delivers a list that provides all found words with file name where they are found
    
    Returns:
		tuple of word and file list:
			word_list 	list of lists that contain stemmed words including the file name (number) where it occurs
			file_list	list of all file names 
        
    """
    
    word_list = []
    file_list = []
    stemmer = PorterStemmer()
    for subdir, dirs, files in os.walk(training_path):
		# read in every file in the training data folder
        filecount = 0
        for file in files:
            filecount += 1
            f = open(training_path+file, 'r')

            ##
            # TODO: handle xml
            ##

            for line in f:
                # tokenize line in sentences
                sentences = nltk.sent_tokenize(line)
                for sentence in sentences:
                    # tokenize sentences in words
                    words = nltk.word_tokenize(sentence)
                    for word in words:
                        # skip all words that contain just one item of punctuation
                        if word in string.punctuation: 
                            continue
                        # add stemmed word and file name containing it to word_list
                        word_list.append([word_stemming(word, stemmer), int(file)])

            file_list.append(str(file))
    f.close()
    return (word_list, file_list)


def word_stemming(old_word, stemmer):
    return stemmer.stem(old_word.lower())


def consolidate_list(sort_word_list, file_list, postings_file, dictionary_file):
    """
    consolidates the given word_list 
	by summing up same words and uniting their list of occurrences. 
	writes result to provided dictionary and postings file
    furthermore: calculation of normalization factors (lengths) per document number
            written down as last line in postings_file
	
	Arguments:
		sort_word_list		alphabetically sorted double list of words and their occurrence in files
		file_list			list that contains all file names of the data set
		postings_file		file name + path for posting list results
							will contain all postings belonging to the different dictionary entries
		dictionary_file		file name + path for dictionary results
							will contain a dictionary that is written down in a file 
    """
    
    f = open(postings_file, 'w')
    d = open(dictionary_file, 'w')

    line_index = 0
    current_word = ""
    current_word_freq = {}
    # dictonary for storing the length of the documents
    doc_lengths = {}
    for word, fileNo in sort_word_list:
		# first case - file number can be added to posting list anyway
        if current_word == "":
            current_word = word
            current_word_freq[fileNo] = 1
		# word occurs again - add file number to current posting list
        elif word == current_word:
            if not fileNo in current_word_freq.keys():
                current_word_freq[fileNo] = 1
            else:
                current_word_freq[fileNo] += 1
		# a new word
        else:
            # write previous word including frequency of posting entries and pointer to posting file line to dictionary
            write_to_dict(d, current_word, current_word_freq, line_index)
            # write posting list to posting file
            doc_lengths = write_to_postings(f, current_word_freq, doc_lengths)
			#start new list for new word
            current_word_freq = {}
            current_word_freq[fileNo] = 1
            current_word = word
            line_index += 1

    # write last word and postings to files        
    write_to_dict(d, current_word, current_word_freq, line_index)
    doc_lengths = write_to_postings(f, current_word_freq, doc_lengths)

    write_all_files(f, file_list)
    write_doc_lengths(f, doc_lengths)

    f.close()
    d.close()


def write_to_dict(d, current_word, current_word_freq, line_index):
    d.write(current_word+" "+str(len(current_word_freq))+" "+str(line_index)+"\n")


def write_to_postings(f, current_word_freq, doc_lengths):
    """
    writes the postings to the posting file and updates the document length vector with the new values

    Arguments:
        f                   postings file
        current_word_freq   the document frequency for the current term accessable by document number
        doc_lengths: list   containing normalization factors accessable by document number
    
    Returns:
        doc_lengths: the updated length vector
    """
    current_postings_keys = sorted(current_word_freq.iterkeys())
    for postingsFileNo in current_postings_keys:
        log_tf = get_tf_value(current_word_freq, postingsFileNo)
        f.write(str(postingsFileNo)+" "+str(log_tf)+" ")
        # update the document length vector with the new tf value
        doc_lengths = add_value_to_doc_length(doc_lengths, postingsFileNo, log_tf)
    f.write("\n")
    return doc_lengths


def get_tf_value(current_word_freq, postingsFileNo):
    """
        applies sublinear tf scaling
    """
    if current_word_freq[postingsFileNo] == 0:
        return 0
    else:
        return 1 + math.log10(current_word_freq[postingsFileNo])


def add_value_to_doc_length(doc_lengths, docNo, tf):
    """
    adds the new tf value to the power of two to the calculation of the document length

    Arguments:
        doc_lengths: list containing normalization factors accessable by document number
        docNo: document number
        tf: the log-tf value to be added
    
    Returns:
        doc_lengths: the updated length vector
    """
    if docNo in doc_lengths.keys():
        doc_lengths[docNo] += math.pow(tf, 2)
    else:
        doc_lengths[docNo] = math.pow(tf, 2)
    return doc_lengths


def write_all_files(f, file_list):
    """
        write all file names as universal set to file
    """
    for fileNo in file_list:
        f.write(str(fileNo) + " ")
    f.write("\n")


def write_doc_lengths(f, doc_lengths):
    """
        write document lengths to file
    """
    for i in range(max(doc_lengths.keys()) + 1):
    # for length_val in sorted(doc_lengths.keys()):
        if i in doc_lengths.keys():
            sqrt_length_val = math.sqrt(doc_lengths[i])
            f.write(str(sqrt_length_val) + " ")
        else:
            f.write(str(0) + " ")


def calculate_length_vector(dictionary, postings_file, line_positions, all_postings, n):
    """
    calculates the normalization factor for each document

    Arguments:
        dictionary:     dict structure for all terms (in memory) 
        postings_file:  contains all postings belonging to the different dictionary entries
        line_positions
        all_postings:
        n:              number of documents in training data

    Returns:
        length_vector   array that contains the normalization factor for each document accessable by document number
    """

    # list of n normalization factors initialized with 0 - length: highest document number
    length_vector = [0 for y in range(int(all_postings[n-1])+1)]

    keys = dictionary.keys()

    # loop through all terms in the dictionary
    for i in range(len(dictionary)):
        term = keys[i]
        term_postings = get_postings(term, dictionary, postings_file, line_positions)

        # reorder tf in postings to length vector, parallelly calculating length
        for (docNo, tf) in term_postings:
            length_vector[docNo] += math.pow(tf,2)

    length_vector = [math.sqrt(x) for x in length_vector]

    return length_vector


def usage():
    print "usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file"


######################
# MAIN
######################

training_path = "patsnap-corpus/"
dictionary_file = "dictionary.txt"
postings_file = "postings.txt"

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError, err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-i':
        path_to_training = a
    elif o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    else:
        assert False, "unhandled option"

indexing(training_path, postings_file, dictionary_file)
