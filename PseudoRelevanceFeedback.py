import nltk
import math
import operator
from collections import Counter
import xml.etree.ElementTree as et
import text_processing
from nltk.stem.porter import *
from IPC import IPC

class PseudoRelevanceFeedback:
    '''
        Class for making new search with Pseudo Relevance Feedback
    '''

    def __init__(self, dictionary, postings_file, line_positions, training_path, n):
        self.dictionary = dictionary
        self.postings_file = postings_file
        self.line_positions = line_positions
        self.training_path = training_path
        self.n = n

    def generate_new_query_topk(self, old_results, no_of_terms, old_query_str):
        """
            public method for finding new query words from the top k retreived files

            Arguments:
                old_results     the k top ranked documents to retreive new query words from
                no_of_terms     the number of new query terms that should be generated
                old_query_str   the original query for the first search

            Return:
                query_weights    dictionary of the words with highest weight from the documents
        """

        content = self.__get_document_content(old_results)
        top_query_terms, old_query_list = self.__get_new_terms(content, old_query_str, no_of_terms)
        query_weights = self.__get_query_tf(top_query_terms, old_query_list)
        return query_weights

    def generate_new_query_topIPC(self, old_results, no_of_terms, old_query_str, patent_info):
        """
            public method for finding new query words from the top IPC subclass

            Arguments:
                old_results     the k top ranked documents to retreive new query words from
                no_of_terms     the number of new query terms that should be generated
                old_query_str   the original query for the first search

            Return:
                query_weights    dictionary of the words with highest weight from the documents
        """
        best_IPC = self.__get_best_IPC(old_results, patent_info)
        
        patentNos = best_IPC.getPatents(patent_info)

        patentNos_dict = []
        for patent in patentNos:
            patentNos_dict.append((patent,0))
            
        content = self.__get_document_content(patentNos_dict)
        # query_weights = self.__get_new_terms_IPC(content, old_query_str, no_of_terms)
        top_query_terms, old_query_list = self.__get_new_terms(content, old_query_str, no_of_terms)
        query_weights = self.__get_query_tf(top_query_terms, old_query_list)

        return query_weights

    def __get_new_terms_IPC(self, content, old_query_str, no_of_terms):
        old_query_list = self.__tokenize_string(old_query_str)
        new_query_list = self.__tokenize_string(content)
        # term_list = [x for x in new_query_list if x not in old_query_list]

        term_count = Counter(new_query_list)

        #top_query_terms = self.__get_top_terms(term_count, no_of_terms)

        stemmer = PorterStemmer()
        query_weights = {}
        for term, count in term_count.items():
            stemmed_term = stemmer.stem(term)
            weight = self.__get_weight_query_term(stemmed_term, count)
            query_weights[term] = weight

        top_query_terms = [x[0] for x in sorted(query_weights.items(), key=operator.itemgetter(1), reverse=True)][:no_of_terms]

        # get weights of top terms
        top_query_terms_weights = {}
        for term in top_query_terms:
            top_query_terms_weights[term] = query_weights[term]

        return top_query_terms_weights



        return (top_query_terms, old_query_list)

    def __get_best_IPC(self, old_results, patent_info):
        """
        returns IPC subclass that occures most within given patent numbers (in old_results)
        """
        ipcSubclasses = {}
        for doc_name, score in old_results:
            # get IPC subclass of patent
            ipcSub = IPC(patent_info[doc_name][2]).subclass()

            # count occurances of subclasses
            if ipcSub in ipcSubclasses.keys():
                ipcSubclasses[ipcSub] += 1
            else:
                ipcSubclasses[ipcSub] = 1

        # get best IPC
        best_subclass = max(ipcSubclasses.iteritems(), key=operator.itemgetter(1))[0]

        return best_subclass

    def __get_document_content(self, old_results):
        """
            Get a combined string with all words from both the title and abstract of the received documents. 

            Arguments:
                old_results     the k top ranked documents to retreive new query words from

            Returns:
                content         the combined string of all words from the documents
        """
        content = ''
        for doc_name, score in old_results:
            path = self.training_path + doc_name + '.xml'
            tree = et.parse(path)
            root = tree.getroot()

            for child in root:
                if child.get('name') == 'Title':
                    content += child.text.encode('utf-8') + ' '

                if child.get('name') == 'Abstract':
                    content += child.text.encode('utf-8') + ' '

        # remove non utf-8 characters, http://stackoverflow.com/a/20078869
        content = re.sub(r'[^\x00-\x7F]+',' ', content)

        return content

    def __get_new_terms(self, content, old_query_str, no_of_terms):
        """
        Tokenize the string of words from the documents and compute the frequency of each word in the query list

        Arguments:
            content         string of words from the documents
            old_query_str   the original query for the first search

        Returns: 
            top_query_terms list of the words from the documents with the highest weights
        """
        old_query_list = self.__tokenize_string(old_query_str)
        new_query_list = self.__tokenize_string(content)
        # term_list = [x for x in new_query_list if x not in old_query_list]

        term_count = Counter(new_query_list)

        top_query_terms = self.__get_top_terms(term_count, no_of_terms)
        return (top_query_terms, old_query_list)

    def __tokenize_string(self, the_string):
        """
            Tokenize and stem a string and remove punctuation and stopwords
        """
        word_list = []
        sentences = nltk.sent_tokenize(the_string)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                normalized = text_processing.normalize(word)
                if normalized is not None:
                    word_list.append(normalized)
                
        return word_list

    def __get_top_terms(self, term_count, no_of_terms):
        """
            Calculate the weight for all words and return the ones with the highest tf-idf

            Arguments:
                term_count      dictionary with the words and the their frequency in the query
                no_of_terms     the number of new query terms that should be generated

            Returns:
                top_query_terms list of the words with the highest tf-idf
        """
        stemmer = PorterStemmer()
        query_weights = {}
        for term, count in term_count.items():
            stemmed_term = stemmer.stem(term)
            weight = self.__get_weight_query_term(stemmed_term, count)
            query_weights[term] = weight

        top_query_terms = [x[0] for x in sorted(query_weights.items(), key=operator.itemgetter(1), reverse=True)][:no_of_terms]
        # query_string = ' '.join(top_query_terms)
        return top_query_terms

    def __get_weight_query_term(self, term, term_count):
        """
        calculates the weight of a term in a query by the pattern tf.idf
        if term is not defined in dictionary => weight_query = 0

        Arguments:
            term        term in query to calculate for
            term_count  the number of times the term occures in the query

        Returns:
            weight of the query term
        """

        # calculate tf in query
        log_tf = self.__compute_tf(term_count)

        if term in self.dictionary.keys():
            (freq, postings_line) = self.dictionary[term]
            # calculate idf
            idf = math.log10(float(self.n)/float(freq))
            return log_tf * idf
        else:
            return 0

    def __get_query_tf(self, new_query_terms, old_query_terms):
        """
            Set tf values for the new terms to 0.5 and for the old ones to 1.5
            Reference: http://ceur-ws.org/Vol-1176/CLEF2010wn-ImageCLEF-Larson2010.pdf
        """
        old_query_count = Counter(old_query_terms)

        query_weights = {}
        for term in new_query_terms:
            query_weights[term] = 0.5
        for term in old_query_terms:
            if term in new_query_terms:
                query_weights[term] = self.__compute_tf(old_query_count[term])*1.5
            else:
                query_weights[term] = self.__compute_tf(old_query_count[term])
        return query_weights

    def __get_query_tf_test(self, new_query_terms, old_query_terms):
        """
            Set tf values for the new terms to 0.5 and for the old ones to 1.5
            Reference: http://ceur-ws.org/Vol-1176/CLEF2010wn-ImageCLEF-Larson2010.pdf
        """
        old_query_count = Counter(old_query_terms)

        query_weights = {}
        for term in new_query_terms:
            query_weights[term] = 1000
        for term in old_query_terms:
            if term in new_query_terms:
                query_weights[term] = self.__compute_tf(old_query_count[term])*1.5
            else:
                query_weights[term] = self.__compute_tf(old_query_count[term])
        return query_weights

    def __compute_tf(self, term_count):
        return 1 + math.log10(term_count)

            

