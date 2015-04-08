import nltk
import string
import math
import operator
from collections import Counter
import xml.etree.ElementTree as et
from nltk.stem.porter import *

DBG_USE_STOPS = True

class PseudoRelevanceFeedback:
    '''
        Class for making new search with Pseudo Relevance Feedback
    '''

    __stops = nltk.corpus.stopwords.words('english')

    def __init__(self, old_results, dictionary, postings_file, line_positions):
        self.old_results = old_results
        self.dictionary = dictionary
        self.postings_file = postings_file
        self.line_positions = line_positions

    # Todo: Clean up and split into separate functions
    def generate_new_query(self, training_path, n):
        stemmer = PorterStemmer()

        content = ''
        for (doc_name, score) in self.old_results:
            tree = et.parse(training_path + doc_name + '.xml')
            root = tree.getroot()

            for child in root:
                if child.get('name') == 'Title':
                    content += child.text.encode('utf-8') + ' '

                if child.get('name') == 'Abstract':
                    content += child.text.encode('utf-8') + ' '

        query_list = []
        sentences = nltk.sent_tokenize(content)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                # skip all words that contain just one item of punctuation
                if word in string.punctuation or (DBG_USE_STOPS and word in self.__stops): 
                    continue
                # add stemmed word the query_list
                query_list.append(stemmer.stem(word.lower()))

        query = Counter(query_list)

        query_weights = {}
        for term, count in query.items():
            weight = self.__get_weight_query_term(term, count, n)
            query_weights[term] = weight

        best_query_terms = [x[0] for x in sorted(query_weights.items(), key=operator.itemgetter(1), reverse=True)][:30]
        
        return best_query_terms

    #TODO: Think about if this is the correct way to calculate the weight in this case!
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

            

