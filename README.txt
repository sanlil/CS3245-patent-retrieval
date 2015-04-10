A0135280M, A0135622M and A0134784X
mail: a0135280@u.nus.edu, a0135622@u.nus.edu, a0134784@u.nus.edu

== General Notes about this assignment ==

== General utility ==

Our normalization function for tokens is located in text_processing.py. It
lowercases words, stops them and stems them. This function is used throughout
the other files, for indexing, querying and generally treating input text.

== Architecture INDEXING ==

files: index.py
generated files: dictionary.txt, postings.txt, patent_info.txt

Indexing is done by reading each file in the patsnap-corpus and extracting the
information that we consider as useful for the seaching process:
We read in the title and abstract, which we use for producing the dictionary
and postings file as well as further fields like Publication Year, # Cited by, 
IPC Primary, 1st Inventor. Those fields are saved in the file patent_info.txt
file, which is structured in rows identified by the patent id and with
columns delimited by vertical pipes ("|").

The free-text regions of the patents (title and abstract) are read in 
lexicographical order using the built-in ElementTree XML library. A python
dictionary stores the postings for every term that is encountered in the
patents, and is populated as each patent is processed. Positional information
is saved in each posting. The distinction between title and abstract is removed
in the index.

Once all the patents are processed, the dictionary and postings are written
to file. The dictionary.txt file contains one dictionary entry per line,
consisting of the word being indexed, its df and a line pointer to the
postings file, indicating on which line of postings.txt can the postings list
for this term be found. All these values are separated by spaces. On the last
line of the dictionary file, the path to the patent corpus is written, to be
accessible during search for various purposes detailed below.

The postings.txt file contains one postings list per line. Each posting is made
up of four elements: a patent ID, a log-tf, a tf and a position list. The
position list indicates at which indices of the patent can the indexed term be
found. It contains exactly tf elements. The very last line of the postings file
contains a list of documents and their normalization factor. Each of these
entries is simply a patent ID followed by a float value, followed by another
entry. The normalization factor is computed during the first processing phase.

== Architecture SEARCHING ==

files: search.py
generated files: output.txt

Regarding the search we tried several different approaches to come up with our
current / final result. In the following we want to point out our final
structure as well as our thoughts, experiments and outcomes:


~ Final result:

The query XML file is parsed again using "ElementTree". A standard free-text
query is constructed, as well as several phrasal queries (more details below).
The free-text query ignores the difference between the title and the text,
since manual research has shown that the words used in the titles of the
patents often do not describe the patent very well.

The dictionary and patent info files are read into memory. In both cases a
python dictionary is used (keyed by term for the dictionary and by patent
for the patent info). 

The same is done with the normalization vectors (last line of the postings 
file) and patent corpus location. Finally, we also compute the byte offsets
of the lines of the postings file, to be able to seek it correctly and avoid
loading the whole file in memory.

Once this is done, the phrasal queries are run. A phrasal query works by first
obtaining the set of documents that contain all the tokens in query. Then,
the postings for each document and term are retrieved and checked against each
other to verify whether a document contains the phrasal query. If yes, then it
is retained. For scoring, we use the cosine similarity between the document
and the phrasal query (there treated as a bag of words), multiplied by the
length of the phrasal query. This is to give phrasal queries a bonus over
simple free-text queries.

We then apply the Vector Space Model to generate a list of patents and their 
scores ordered in descending order of score. We then use pseudo-relevance
feedback on both the top documents and top IPC. Relevant terms are harvested
from a set of top documents, and from document that have a relevant IPC. A
new query is constructed from those terms and the original query. 
Pseudo-relevant terms are given a weight of 0.5, and original query terms see
their weight increased by 50%. This new query is ran, and the scores from it
are added to the phrasal query scores to yield the final score. All results are
written down to the output file.


~ Thoughts, experiments and outcomes:

We started with a vector space model implemented in VectorSpaceModel.py as a 
base. The goal was then to add additional methods and techniques to improve our
outcomes. The recall was always good as all relevant documents could be
retrieved. Some of them were ranked very low though (near the end of 2000 
documents, for example). We tried to improve the ranking of the VSM scores with
several methods detailed below.

Using stop words;
    This improved the ranking of several relevant documents by preventing other
    irrelevant documents from being retrieved. We went with a standard list of
    English stopwords found in NLTK. We also added some of our own stop words
    (see text_processing.py).
    
Removing words with low idf;
    Similar to using stop words but in a more automated fashion. We did not get
    around to implementing this.

Pseudo-relevance feedback:
	We used two variants of this. The first one takes the top K relevant 
    documents, then finds N new, high-tf.idf words from those documents. 
    The second variant uses the most common IPC in the top K documents to
    obtain other patents to find new words from. The IPC is considered at
    the subclass level. As described above, the new terms are weighted less in
    the second query. This prevents the query from becoming too general. The 
    values for N and K in the code were decided upon with some experimentation
    against the two example queries.
    
Field / zone treatment:
	We though about using the following fields: times the patent was cited, 
    publication date and inventor. The intuitions were as follows: a patent
    cited more often may be more relevant (similar to PageRank, actually),
    more recent patents may be more relevant, and the inventor may produce
    inventions in similar fields that we might be interested in.
    
    Examination of the number of citations and publication date found no 
    correlation between those and relevance. As for the inventor, we did not
    find that an inventor was particularly likely to have more relevant patents
    to his name. Hence we did not use any of these.
    
    We also looked at using the IPC to constrain the search. It did not give
    improved results as most of the top documents were already in the same IPC
    subclass. We also thought about guessing relevant IPCs from the query, but
    this would have been difficult and we had no problem with it anyway.

Query Expansion:
    We thought about using synonyms from wordnet to expand the query, but we 
    found that our problems with ranking relevant documents well would not have
    been solved by simply adding synonyms, so we did not do it.

Positional Index and Phrasal Queries:
	We used phrasal queries generated from the original queries using some
    simple rules (e.g. try to flatten enumerations of actions and agents into
    several phrasal queries). These improved our results somewhat.

---


The main files that we used are built in the following structure:

dictionary_file:	contains a dictionary that is written down in a file 
                    form: "term frequency line-in-postings-file" 
					the line-in-postings-file is meant like a pointer to the specific line in the postings_file
postings_file:  	contains all postings belonging to the different dictionary entries
                    form: "patentNo (1+ log_tf) tf list-of-occurences"
                        the list of occurences is a list of integers giving the positions
                        of the term in the document.
                    LAST LINE: contains information about the length vector / normalization factor per patent
                    form: "patentNo   normalizationFactor"
output_file:    	shows all file numbers that result from the queries
                    form: "file_name <space> file_name <space>..."
patent_info:		contains information about the different patents
					form: "patentNo |  Publication Year  |  # Cited by |  IPC Primary  | 1st Inventor"


== Files included with this submission ==

PYTHON FILES
index.py        			Creates the index consisting of the dictionary, 
                            postings and patent info files.
search.py       			Reads the dictionary file into memory, formats and
                            solves queries with the information in the 
                            postings file.
phrasal_queries.py          Provides functions to perform phrasal queries on
                            the index, and to generate phrasal queries from
                            a free-text query.
VectorSpaceModel.py 		Class to perform VSM scoring on free-text or
                            phrasal queries.
PseudoRelevanceFeedback.py  Class to perform pseudo-relevance feedback as
                            described above (either top K documents or top IPC)
IPC.py 						Class representing an IPC symbol, providing
                            operations to compare IPC symbols, test for
                            membership/hierarchy and obtain patents having a 
                            given IPC.
text_processing.py 			Includes general functionality for applying 
                            case-folding, stemming and stopping

TEXT FILES
dictionary.txt  The dictionary
postings.txt    The postings list
patent_info.txt information extracted from patent corpus structured by patent ID

README.txt      Information about the submission (this file)


== Allocation of work ==

discussion and conception  			A0135280M, A0135622M, A0134784X
basic indexing 						A0135280M, A0135622M
    positional indexing             A0134784X
free-text queries					A0135280M, A0135622M
    phrasal queries                 A0134784X
debug prints of results				A0134784X
vector space model  				A0135280M - main parts of the code produced in previous assignment by A0135280M, A0135622M
pseudo relevance feedback 			A0135280M 
General IPC functionality			A0134784X
    Retrieving patents by IPC       A0135622M
Latent Semantic Analysis			A0135622M (research)
README.txt 							A0135622M, A0134784X


== Statement of individual work ==

Please initial one of the following statements.

[X] We, A0135280M, A0135622M and A0134784X certify that we have followed
the CS3245 Information Retrieval class guidelines for homework assignments.
In particular, we expressly vow that we have followed the Facebook rule in
discussing with others in doing the assignment and did not take notes (digital
or printed) from the discussions.  

[ ] We, A0135280M, A0135622M and A0134784X did not follow the class rules 
regarding homework assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==
* We used the cosine score algorithm that was teached in the lesson.

* We used a solution proposed on this page to get the correct line position used for the seek() method: 
    http://stackoverflow.com/questions/620367/python-how-to-jump-to-a-particular-line-in-a-huge-text-file 
    
* https://docs.python.org/2/reference/datamodel.html

* http://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging

* The idea to weigh new and old terms differently when doing pseudo-relevance
    feedback comes from the following paper: 
    Larson, R. Ray 2010, Blind Relevance Feedback for the ImageCLEF Wikipedia Retrieval Task, http://ceur-ws.org/Vol-1176/CLEF2010wn-ImageCLEF-Larson2010.pdf

* https://docs.python.org/2/library for various libraries, notably bisect.

* The IR textbook by Manning et al., sections 7.2 and 7.3