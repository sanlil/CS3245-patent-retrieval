#!/usr/bin/python
import sys
import getopt
import nltk
import math
import xml.etree.ElementTree as et
from collections import Counter
from VectorSpaceModel import VectorSpaceModel
from PseudoRelevanceFeedback import PseudoRelevanceFeedback
# from IPC import IPC
from collections import Counter
import text_processing

DEBUG_RESULTS = False
PRINT_IPC = False
DBG_USE_STOPS = True
USE_PRF = True

def print_result_info(scores, retrieve, not_retrieve, patent_info):
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
    
    # Print information about the documents that we wanted to retrieve:
    # Which ones were retrieved, which ones were missed, and for those retrieved,
    # where did they appear in our ranking.
    correctly_obtained = wanted & retrieved_docs
    nco = len(correctly_obtained)
    nw = len(wanted)
    print 'Correctly hit %d documents out of %d wanted (%f%%)' % (nco, nw, nco / float(nw) * 100.0)
    print 'Documents missed: %s\n' % ', '.join(wanted - correctly_obtained)
    
    print 'Rankings/IPC of hit documents:\n\t',
    print '\n\t'.join([doc + ':' + ((20 - len(doc)) * ' ') + str(rank) for doc, rank in positional_scores if doc in correctly_obtained])
    
    print ''
    
    # Print information about the documents that we wanted to avoid:
    # Which ones were retrieved, and what were their ranking
    incorrectly_obtained = unwanted & retrieved_docs
    nio = len(incorrectly_obtained)
    nu = len(unwanted)
    print 'Incorrectly retrieved %d documents out of %d to avoid (%f%%)' % (nio, nu, nio / float(nu) * 100.0)
    print 'Rankings of false positives:\n\t',
    print '\n\t'.join([doc + ':' + ((20 - len(doc)) * ' ') + str(rank) for doc, rank in positional_scores if doc in incorrectly_obtained])
    
    if not PRINT_IPC:
        return
    
    print 'IPC subclasses in the top 50 documents: '
    top = Counter([str(IPC(patent_info[doc][2]).subclass()) for doc, score in scores[:50]])
    print top

def print_patent_info(retrieve, not_retrieve, patent_info):
	"""
	print out patent_info of relevant and irrelevant patents
	"""
	# Obtain the documents that we should retrieve and those that we shouldn't
	with open(retrieve) as r, open(not_retrieve) as n:
		wanted = set([l.rstrip() for l in r.readlines()])
		unwanted = set([l.rstrip() for l in n.readlines()])

	publication_years_rel = {}
	print "Relevant documents:"
	for patentNo in wanted:
		publication_year = patent_info[patentNo][0]
		count_citation = patent_info[patentNo][1]
		if publication_year in publication_years_rel.keys():
			publication_years_rel[publication_year] += 1
		else:
			publication_years_rel[publication_year] = 1

		#print patentNo, "\t\t", publication_year, count_citation

	for key in sorted(publication_years_rel.keys()):
		print key, publication_years_rel[key]
	print "---"

	publication_years_irrel = {}
	print "Irrelevant documents:"
	for patentNo in unwanted:
		publication_year = patent_info[patentNo][0]
		count_citation = patent_info[patentNo][1]
		if publication_year in publication_years_irrel.keys():
			publication_years_irrel[publication_year] += 1
		else:
			publication_years_irrel[publication_year] = 1

		#print patentNo, "\t\t", patent_info[patentNo][0], patent_info[patentNo][1]  

	for key in sorted(publication_years_irrel.keys()):
		print key, publication_years_irrel[key]

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
    
    dictionary, training_path = read_dict(dictionary_file)   
    line_positions, last_line_pos = get_line_positions(postings_file);
    patent_info = get_patent_info(patent_info_file)
    length_vector, n = get_length_vector(postings_file, last_line_pos)
    org_query_str = extract_query_words(query_file)
    org_query = process_query(org_query_str)

    # patent_info
    # print_patent_info(retrieve, not_retrieve, patent_info)

    VSM = VectorSpaceModel(dictionary, postings_file, line_positions)
    scores = VSM.get_scores(org_query, last_line_pos, length_vector, n)

    # DEBUG DEBUG
    # print VSM.get_phrasal_score("washing machine", last_line_pos, length_vector, n)
    # return

    if USE_PRF:
        no_of_documents = 10
        no_of_terms = 10
        PRF = PseudoRelevanceFeedback(dictionary, postings_file, line_positions, training_path, n)
        new_query = PRF.generate_new_query(scores[:no_of_documents], no_of_terms, org_query_str)

        scores = VSM.get_scores(new_query, last_line_pos, length_vector, n)
    
    if DEBUG_RESULTS:
        print_result_info(scores, retrieve, not_retrieve, patent_info)
    
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
        if "#" in line:
            training_path = line.split(' ')[1].rstrip()
        else:
            word, freq, postings_line = line.split(' ')
            dictionary[word] = (freq, postings_line)

    d.close()
    return dictionary, training_path

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

def get_length_vector(postings_file, last_line_pos):
    """
        Arguments:
            last_line_pos   the position of the last line in the postings file containing all normalization factors

        Return:
            length_vector   dictionary containing the normalization factor for each document accessable by document name
            n               the number of documents in the training data
    """
    f = open(postings_file, 'r')
    f.seek(last_line_pos)
    postings_list = f.readline().split()
    length_vector = {}
    n = 0
    i = 0
    while i < len(postings_list):
        doc_name = postings_list[i]
        length = postings_list[i+1]
        length_vector[doc_name] = float(length)
        n += 1
        i += 2
    f.close()
    return (length_vector, n)

def extract_query_words(query_file):
    """
    Parse the query xml files and extract the titles and descriptions of the documents

    Arguments:
        query_file  path to the file containing the query

    Returns: 
        content     string of all the query words
    """

    tree = et.parse(query_file)
    root = tree.getroot()

    query_content = ''

    for child in root:
        if child.tag == 'title':
            query_content += child.text.encode('utf-8')

        if child.tag == 'description':
            desc = child.text.encode('utf-8')
            # remove the 4 first words since they always will be: "Relevant documents will describe"
            query_content += ' '.join(desc.split()[4:])

    return query_content

def process_query(query_str):
    """
    Tokenize and stem the query words and compute the frequency of each word in the query list

    Arguments:
        query_str       string of query words

    Returns: 
        query_count     a dictionary with the stemmed words and the its frequency in the query
    """
    
    query_list = []
    sentences = nltk.sent_tokenize(query_str)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:            
            normalized = text_processing.normalize(word)
            if normalized is not None:
                query_list.append(normalized)
            
    # count the frequency of each term
    query_count = Counter(query_list)

    # set the tf value for each term
    query_weight = {}
    for query_term, term_count in query_count.items():
        query_weight[query_term] = 1 + math.log10(term_count)

    return query_weight

def write_to_output_file(output_file, scores):
    """
    writes the scores to output_file
    """
    f = open(output_file, 'w+')
    
    #just for debugging - remove later
    output_path_parts = output_file.split('/')
    output_path_parts[len(output_path_parts)-1] = 'debug_' + output_path_parts[len(output_path_parts)-1]
    debug_output_path = '/'.join(output_path_parts)
    g = open(debug_output_path, 'w+')
    
	# Fred: output has no trailing space, and ends with a newline
    f.write(' '.join([doc_name for doc_name, score in scores]) + '\n')
    
    g.write('\n'.join([doc_name + ' ' + str(score) for doc_name, score in scores])) #just for debugging - remove later	

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
