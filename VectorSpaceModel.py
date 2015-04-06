import math
import operator
from collections import Counter
from nltk.stem.porter import *

class VectorSpaceModel:
    '''
        Class for calculating score with the Vector Space Model
    '''

    def __init__(self, query, dictionary, postings_file, line_positions, last_line_pos):
        self.query = query
        self.dictionary = dictionary
        self.postings_file = postings_file
        self.line_positions = line_positions
        self.last_line_pos = last_line_pos

    def getScores(self):
        length_vector, n = self.get_length_vector()
        scores = self.calculate_cosine_score(length_vector, n)
        return scores

    def get_length_vector(self):
        """
            Return:
                length_vector   dictionary containing the normalization factor for each document accessable by document name
                n               the number of documents in the training data
        """
        f = open(self.postings_file, 'r')
        f.seek(self.last_line_pos)
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

    def calculate_cosine_score(self, length_vector, n):
        """
        computes the cosine scores for a query and all given documents and returns the best results

        Arguments:
            length_vector     dictionary containing the normalization factor for each document accessable by document name
            n                 the number of documents in the training data

        Returns:
            ordered_scores    list of tuples with document names and scores ordered by decreasing score
        """

        scores = {}
        length_query = 0

        query_counts = Counter(self.query)

        for query_term, term_count in query_counts.items():
            weight_query = self.get_weight_query_term(query_term, term_count, n)
            length_query += math.pow(weight_query, 2)
            term_postings = self.get_postings(term)
            all_relevant_documents = []

            for (doc_name, tf) in term_postings:
                all_relevant_documents.append(doc_name)
                weight_d_t = float(tf)
                if doc_name in scores.keys():
                    scores[doc_name] += weight_d_t * weight_query
                else:
                    scores[doc_name] = weight_d_t * weight_query
        
        length_query = math.sqrt(length_query)

        for doc_name in all_relevant_documents:
            # if (length_vector[doc_name]*length_query) != 0:
                scores[doc_name] = scores[doc_name]/(length_vector[doc_name]*length_query)

        # score_list = [x[0] for x in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)]
        ordered_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True);
        return ordered_scores

    def get_postings(self, word):
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
           list of tuples (doc_name, tf) corresponding to the word
        """

        no_of_postings_parameters = 4

        # dictionary contains word
        if word in self.dictionary.keys():
            (freq, postings_line) = self.dictionary[word]
            f = open(self.postings_file, 'r')
            f.seek(self.line_positions[int(postings_line)])
            postings_list = f.readline().split()
            doc_tf_list = []
            count = 0
            while count < len(postings_list):
                doc_name = postings_list[count]
                tf = postings_list[count+1]
                doc_tf_list.append( ( doc_name, float(tf) ) )
                count += no_of_postings_parameters
            f.close()
            return doc_tf_list
        # dictionary does not contain word
        else:
            return []

    def get_weight_query_term(self, term, term_count, n):
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
        log_tf = 1 + math.log10(term_count)

        if term in self.dictionary.keys():
            (freq, postings_line) = self.dictionary[term]
            # calculate idf
            idf = math.log10(float(n)/float(freq))
            return log_tf * idf
        else:
            return 0


