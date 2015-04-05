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


def indexing(training_path, postings_file, dictionary_file, patent_info_file):
    """
    reads the training data and creates a postings and a dictionary file
    
    Arguments:
        training_path       folder containing all training data files to be used
        dictionary_file:    path to file containing a dictionary in the end that is written down in a file 
                            (form: "term frequency line-in-postings-file(-->'pointer')")    
        postings_file:      path to file containing all postings belonging to the different dictionary entries
                            form: "file_name <space> file_name <space>..."
                            the pre-last line contains the names of all given documents in the training data
                            the last line contains the length vector (all normalization factors from 0 until the highest document ID)
        patent_info_file    path to file containing meta data for patents
                            form: "patentNo | 1st Inventor | Cited by Count | IPC Primary | Publication Year"
    """

    word_list = read_training_data(training_path, patent_info_file)
    sorted_word_list = sort_alphabetically(word_list)

    consolidate_list(sorted_word_list, postings_file, dictionary_file)


def sort_alphabetically(word_list):
    return sorted(word_list,key=lambda x: x[0])


def read_training_data(training_path, patent_info_file):
    """
    reads all patents that can be found in the given folder of training data,
	tokenizes their content, stems the words 
	and delivers a list that provides all found words with file name where they are found

    additionally it creates a patent_info_file that contains meta data about the patent

    Arguments:
        training_path
        patent_info_file    path to file containing meta data for patents
    
    Returns:
		word_list 	list containing stemmed words together with the file name where it occurs
        
    """
    pi = open(patent_info_file, 'w')
    word_list = []

    for subdir, dirs, files in os.walk(training_path):
		# read in every file in the training data folder
        count = 0;
        for file in files:
            fileName = file.split('.')[0]
            # if count > 1:
            #     break
            # count += 1;
            tree = et.parse(training_path + file)
            root = tree.getroot()
            title_content = ""
            abstract_content = ""
            patent_info = fileName

            for r in root:
                # extract patent content
                if r.get('name') == 'Title':
                    title_content += r.text.encode('utf-8') + " "

                if r.get('name') == 'Abstract':
                    abstract_content += r.text.encode('utf-8') + " "

                # extract patent info (meta data)
                if r.get('name') == '1st Inventor':
                    patent_info += " | " + r.text.encode('utf-8').strip()

                if r.get('name') == 'Cited By Count':
                    patent_info += " | " + r.text.encode('utf-8').strip()

                if r.get('name') == 'IPC Primary':
                    patent_info += " | " + r.text.encode('utf-8').strip()

                if r.get('name') == 'Publication Year':
                    patent_info += " | " + r.text.encode('utf-8').strip()

            pi.write(patent_info+"\n")

            title_word_list = create_word_patent_list(title_content, fileName, True)
            abstract_word_list = create_word_patent_list(abstract_content, fileName, False)

            word_list.extend(title_word_list)
            word_list.extend(abstract_word_list)

    pi.close()

    return word_list

def create_word_patent_list(content, fileName, isTitle):
    """
        process extracted content by tokenizing it into words, stemming and stripping punctuation.

        Arguments:
            content     a string sequence
            fileName    the file containing the content

        Return:
            word_list   a list containing the word-file name pairs
    """
    word_list = []
    stemmer = PorterStemmer()
    # remove non utf-8 characters, http://stackoverflow.com/a/20078869
    content = re.sub(r'[^\x00-\x7F]+',' ', content)

    sentences = nltk.sent_tokenize(content)
    for sentence in sentences:
        # tokenize sentences in words
        words = nltk.word_tokenize(sentence)
        for word in words:
            # skip all words that contain just one item of punctuation
            if word in string.punctuation: 
                continue
            # add stemmed word and file name containing it to word_list
            word_list.append([word_stemming(word.encode('utf-8'), stemmer), fileName, isTitle, (not isTitle)])

    return word_list


def word_stemming(old_word, stemmer):
    return stemmer.stem(old_word.lower())


def consolidate_list(sorted_word_list, postings_file, dictionary_file):
    """
    consolidates the given word_list 
	by summing up same words and uniting their list of occurrences. 
	writes result to provided dictionary and postings file
    furthermore: calculation of normalization factors (lengths) per document number
            written down as last line in postings_file
	
	Arguments:
		sorted_word_list	alphabetically sorted double list of words and their occurrence in files
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
    for word, fileName, isTitle, isAbstract in sorted_word_list:
		# first case - file number can be added to posting list anyway
        if current_word == "":
            current_word = word
            current_word_freq[fileName] = 1
            current_word_isTitle = isTitle
            current_word_isAbstract = isAbstract
		# word occurs again - add file number to current posting list
        elif word == current_word:
            if not fileName in current_word_freq.keys():
                current_word_freq[fileName] = 1
            else:
                current_word_freq[fileName] += 1

            current_word_isTitle = current_word_isTitle or isTitle
            current_word_isAbstract = current_word_isAbstract or isAbstract
		# a new word
        else:
            # write previous word including frequency of posting entries and pointer to posting file line to dictionary
            write_to_dict(d, current_word, current_word_freq, line_index)
            # write posting list to posting file
            doc_lengths = write_to_postings(f, current_word_freq, doc_lengths, current_word_isTitle, current_word_isAbstract)
			#start new list for new word
            current_word_freq = {}
            current_word_freq[fileName] = 1
            current_word_isTitle = isTitle
            current_word_isAbstract = isAbstract
            current_word = word
            line_index += 1

    # write last word and postings to files        
    write_to_dict(d, current_word, current_word_freq, line_index)
    doc_lengths = write_to_postings(f, current_word_freq, doc_lengths, current_word_isTitle, current_word_isAbstract)

    write_doc_lengths(f, doc_lengths)

    f.close()
    d.close()


def write_to_dict(d, current_word, current_word_freq, line_index):
    d.write(current_word+" "+str(len(current_word_freq))+" "+str(line_index)+"\n")


def write_to_postings(f, current_word_freq, doc_lengths, current_word_isTitle, current_word_isAbstract):
    """
    writes the postings to the posting file and updates the document length vector with the new values

    Arguments:
        f                   postings file
        current_word_freq   the document frequency for the current term accessable by document number
        doc_lengths: list   containing normalization factors accessable by document number
        current_word_isTitle
        current_word_isAbstract
    
    Returns:
        doc_lengths: the updated length vector
    """
    current_postings_keys = sorted(current_word_freq.iterkeys())
    for docName in current_postings_keys:
        log_tf = get_tf_value(current_word_freq, docName)
        f.write(str(docName)+" "+str(log_tf)+" "+booleanToNo(current_word_isTitle)+" "+booleanToNo(current_word_isAbstract)+" ")

        # update the document length vector with the new tf value
        doc_lengths = add_value_to_doc_length(doc_lengths, docName, log_tf)
    f.write("\n")
    return doc_lengths

def booleanToNo(boolean_value):
    if boolean_value:
        return "1"
    else:
        return "0"


def get_tf_value(current_word_freq, docName):
    """
        applies sublinear tf scaling
    """
    if current_word_freq[docName] == 0:
        return 0
    else:
        return 1 + math.log10(current_word_freq[docName])


def add_value_to_doc_length(doc_lengths, docName, tf):
    """
    adds the new tf value to the power of two to the calculation of the document length

    Arguments:
        doc_lengths: dictionary containing normalization factors accessable by document number
        docNo: document number
        tf: the log-tf value to be added
    
    Returns:
        doc_lengths: the updated length vector
    """
    if docName in doc_lengths.keys():
        doc_lengths[docName] += math.pow(tf, 2)
    else:
        doc_lengths[docName] = math.pow(tf, 2)
    return doc_lengths


def write_doc_lengths(f, doc_lengths):
    """
        write document lengths to file
    """
    count = 0;
    for subdir, dirs, files in os.walk(training_path):
        for fileName in files:
            # if count > 1:
            #     break
            # count += 1
            fileName = fileName.split('.')[0]
            sqrt_length_val = math.sqrt(doc_lengths[fileName])
            f.write(fileName + " " + str(sqrt_length_val) + " ")


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
