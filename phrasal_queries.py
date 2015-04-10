import math
import nltk
import string
from collections import Counter
from nltk.stem.porter import *
from math import sqrt
from bisect import bisect_left

DBG_USE_STOPS = True
stops = set(nltk.corpus.stopwords.words('english'))

def generate_phrasal_queries(title, description):
    """
    Generate some phrasal queries from a given query title and description.
    """
    pass

def list_contains(a, x):
    """
    Check whether the sorted list a contains the element x.
    https://docs.python.org/2/library/bisect.html#searching-sorted-lists
    """
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return True
    return False

def get_documents_with_phrase(query_terms, individual_postings, combined_postings):
    """
    Extract the documents containing the given phrase (query_terms) from
    the postings of all those terms together (combined_postings).
    
    Algorithm from the set of slides "Query Languages" by Raymond Mooney,
    University of Texas at Austin (https://www.cs.utexas.edu/~mooney/ir-course/slides/QueryLanguages.ppt)
    """
    retrieved = set()
    for doc, _, __ in combined_postings:
        positions = {}
        for query_term in query_terms:
            if query_term in positions:
                continue
            positions[query_term] = get_positions(individual_postings[query_term], doc)
        
        # An optimization in Mooney's slides is to use the smallest list of positions.
        # We're not bothering with that here.
        for pos in positions[query_terms[0]]:
            position_not_found = False # Were we unable to find a correct position for one of the query terms?
            for i, query_term in enumerate(query_terms[1:]):
                required = pos + i + 1                    
                if not list_contains(positions[query_term], required):
                    position_not_found = True
                    break
            # If we found a correct position for all the query terms, this document contains the phrase
            if not position_not_found:
                retrieved.add(doc)
                # No need to check any further positions
                break
    return retrieved

def get_positions(postings, document):
    """
    Get the list of positions for this document in the given postings list.
    """
    
    # Basically binary search through the postings for our patent ID
    if len(postings) == 0:
        return None
    
    mid = len(postings) / 2
    if postings[mid][0] == document:
        return postings[mid][2]
    elif postings[mid][0] > document:
        return get_positions(postings[:mid], document)
    else:
        return get_positions(postings[mid+1:], document)

def get_phrasal_postings(query_terms, VSM):
    """
    Retrieve a postings list for the given phrasal query. The
    postings list will contain the documents that contain all
    the given query terms.
    """
    
    query_set = list(set(query_terms)) # Duplicates are not important here so get rid of them
    merged_postings = VSM.get_postings(query_terms[0], True)
    positional_postings = {query_terms[0]: merged_postings}
    
    for query_term in query_terms[1:]:
        fresh_postings = VSM.get_postings(query_term, True)
        positional_postings[query_term] = fresh_postings
        
        merged_postings = merge_postings(merged_postings, fresh_postings)
        
    return (positional_postings, merged_postings)

def merge_postings(a, b):
    """
    Merge the postings lists a and b.
    Reused from A0134784X's homework 2
    """
    skip1 = int(sqrt(len(a)))
    skip2 = int(sqrt(len(b)))
    
    intersection = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] == b[j][0]:
            intersection.append(a[i])
            i += 1
            j += 1
        elif a[i][0] < b[j][0]:
            i += try_skip(a, i, b, j, skip1)
        else:
            j += try_skip(b, j, a, i, skip2)
            
    return intersection

def try_skip(us, p_us, other, p_other, stride):
    """
    Determine whether to skip in the list iteration or not.
    Also reused from A0134784X's homework 2
    """
    if p_us % stride == 0 and p_us + stride < len(us) and us[p_us + stride][0] < other[p_other][0]:
        return stride
    else:
        return 1

def process_phrasal(phrase):
    """
    Tokenize and stem the phrase words and compute the frequency of each word in the query list

    Arguments:
        phrase           string representing the phrasal query

    Returns: 
        A tuple containing the tokenized sentence and a counter object containing the counts
        for the terms in the query.
    """
    stemmer = PorterStemmer()
    
    query_list = []
    sentences = nltk.sent_tokenize(phrase)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:
            # skip all words that contain just one item of punctuation
            if word in string.punctuation or (DBG_USE_STOPS and word in stops): 
                continue
            # add stemmed word the query_list
            query_list.append(stemmer.stem(word.lower()))

    # count the frequency of each term
    query_count = Counter(query_list)
    return (query_list, query_count)
