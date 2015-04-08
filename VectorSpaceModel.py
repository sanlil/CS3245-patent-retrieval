import math
import nltk
import string
import operator
from collections import Counter
from nltk.stem.porter import *

DBG_USE_STOPS = True

class VectorSpaceModel:
    '''
        Class for calculating score with the Vector Space Model
    '''
    
    __stops = nltk.corpus.stopwords.words('english')

    def __init__(self, dictionary, postings_file, line_positions):
        self.dictionary = dictionary
        self.postings_file = postings_file
        self.line_positions = line_positions

    def getScores(self, query, last_line_pos, length_vector, n):
        """
            public class for calculating scores with the vector space model

            Return:
                scores  list of tuples with document names and scores ordered by decreasing score
        """
        query_count = self.__process_query(query)
        scores = self.__calculate_cosine_score(query_count, length_vector, n)
        return scores

    def __process_query(self, query):
        """
        Tokenize and stem the query words and compute the frequency of each word in the query list

        Arguments:
            query           string of query words

        Returns: 
            query_count     a dictionary with the stemmed words and the its frequency in the query
        """
        stemmer = PorterStemmer()
        
        query_list = []
        sentences = nltk.sent_tokenize(query)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                # skip all words that contain just one item of punctuation
                if word in string.punctuation or (DBG_USE_STOPS and word in self.__stops): 
                    continue
                # add stemmed word the query_list
                query_list.append(stemmer.stem(word.lower()))

        # count the frequency of each term
        query_count = Counter(query_list)
        return query_count

    def __calculate_cosine_score(self, query, length_vector, n):
        """
        computes the cosine scores for a query and all given documents and returns the scores for relevant documents

        Arguments:
            query           dictionary containing query words and the number of times it occures in the query            
            length_vector   dictionary containing the normalization factor for each document accessable by document name
            n               the number of documents in the training data

        Returns:
            ordered_scores  list of tuples with document names and scores for relevant documents ordered by decreasing score
        """

        scores = {}
        length_query = 0

        for query_term, term_count in query.items():
            weight_query = self.__get_weight_query_term(query_term, term_count, n)
            length_query += math.pow(weight_query, 2)
            term_postings = self.__get_postings(query_term)
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
            scores[doc_name] = scores[doc_name]/(length_vector[doc_name]*length_query)

        # score_list = [x[0] for x in sorted(scores.items(), key=operator.itemgetter(1), reverse=True)]
        ordered_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True);
        return ordered_scores

    def __get_postings(self, word):
        """
        looks up a word in the given dictionary 
        and returns all postings that belong to that word

        Arguments:
            word            term that is meant to be look up

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

    def __get_weight_query_term(self, term, term_count, n):
        """
        calculates the weight of a term in a query by the pattern tf.idf
        if term is not defined in dictionary => weight_query = 0

        Arguments:
            term        term in query to calculate for
            term_count  the number of times the term occures in the query
            n           number of documents in training data

        Returns:
            weight of the query term
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


