import nltk
import string
import math
from collections import Counter
from bisect import bisect_left
import text_processing

__action_words = set(['with', 'using', 'through', 'means'])
__complementizers = set(['to'])

def generate_phrasal_queries(title, description):
    """
    Generate some phrasal queries from a given query title and description.
    """
    p_title, _ = process_phrasal(title, False)
    p_desc, _ = process_phrasal(description, False)
    
    # For the title, we want some subsets of length 2, 3 plus the title itself
    queries = [' '.join(p_title)]
    for i in range(0, len(p_title) - 2):
        queries.append(' '.join(p_title[i:i+3]))
    for i in range(0, len(p_title) - 1):
        queries.append(' '.join(p_title[i:i+2]))
    
    tagged_desc = nltk.pos_tag(p_desc)
    
    for i, (tok, pos) in enumerate(tagged_desc):
        if tok in __action_words:
            queries.extend(generate_queries_from_action(tagged_desc, i))
        if pos == 'TO':
            queries.extend(generate_queries_from_to(tagged_desc, i))
    
    return queries

def generate_queries_from_to(tagged_desc, where):
    outstanding_terms = []
    i = where
    
    while i > 0:
        i -= 1
        pos = tagged_desc[i][1]
        tok = tagged_desc[i][0]
        if tok == '.' or tok in __action_words:
            break
        if pos.startswith('NN'):
            outstanding_terms.append(tok)
            
        # e.g. "This robot works by means of cheating to win the game"
        # cheating is verb gerund form -> VBG
        if pos == 'VBG':
            outstanding_terms.append(tok)
    
    following_verbs = []
    following_nouns = []
    
    while i < len(tagged_desc) - 1:
        i += 1
        pos = tagged_desc[i][1]
        tok = tagged_desc[i][0]
        if tok == '.':
            break
        
        if pos.startswith('NN'):
            following_nouns.append(tok)
        if pos.startswith('V'):
            following_verbs.append(tok)
    
    all = []       
    for out in outstanding_terms:
        for v in following_verbs:
            all.append(out + ' ' + v)
            for n in following_nouns:
                all.append(out + ' ' + v + ' ' + n)
    
    return all
    
    
def generate_queries_from_action(tagged_desc, where):
    """
    Given the position of an action word ("using", "with", etc.),
    try to generate some queries based on what's around it.
    """
    outstanding_nouns = []
    actions = []
    i = where
    
    # Try and determine the action to do here
    # Note: this will fail to capture the meaning of:
    # DO_A or DO_B [on] X using...
    # we'll get "DO_A using..." and "DO_B X using..."
    while i > 0:
        i -= 1
        pos = tagged_desc[i][1]
        tok = tagged_desc[i][0]
        if tok == '.': # Went up to the beginning of a sentence
            break
        
        if pos.startswith('NN'):
            outstanding_nouns.append(tok)
            
        if pos.startswith('V'):
            actions.append(tok)
            while outstanding_nouns != []:
                actions.append(tok + ' ' + outstanding_nouns.pop())
    
    # This would happen with a sentence of the form
    # "The foo works by means of using bar to baz"
    if actions == []:
        return []
        
    qualified_actions = []
    i = where
    while i < len(tagged_desc) - 1:
        i += 1
        pos = tagged_desc[i][1]
        tok = tagged_desc[i][0]
        if tok == '.': 
            break
        
        if pos.startswith('NN'):
            qualified_actions.extend([action + ' ' + tok for action in actions])
    
    # Want some actual phrasal queries here
    return [s for s in actions + qualified_actions if s.find(' ') != -1]
        
   
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
    skip1 = int(math.sqrt(len(a)))
    skip2 = int(math.sqrt(len(b)))
    
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

def process_phrasal(phrase, normalize=True):
    """
    Tokenize and stem the phrase words and compute the frequency of each word in the query list

    Arguments:
        phrase           string representing the phrasal query

    Returns: 
        A tuple containing the tokenized sentence and a counter object containing the counts
        for the terms in the query.
    """
    
    query_list = []
    sentences = nltk.sent_tokenize(phrase)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if normalize:
            for word in words:
                normalized = text_processing.normalize(word)
                if normalized is not None:
                    query_list.append(normalized)
        else:
            query_list.extend(words)

    # count the frequency of each term
    query_count = Counter(query_list)
    return (query_list, query_count)
