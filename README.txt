A0135280M, A0135622M and A0134784X
mail: a0135280@u.nus.edu, a0135622@u.nus.edu, a0134784@u.nus.edu

== General Notes about this assignment ==

== Architecture INDEXING ==

files: index.py
generated files: dictionary.txt, postings.txt, patent_info.txt

Indexing is done by reading each file in the patsnap-corpus and extracting the information that we consider as useful for the seaching process:
We read in title and abstract which we use for producing dictionary and postings file as well as further tags like Publication Year, # Cited by, IPC Primary, 1st Inventor that are saved in a patent_info file.

We start by reading in the training data patents in xml-format which is supported by the library "ElementTree". 
Both title and abstract are tokenized and processed in the same way and added to a list of words that saves term, patentID and a boolean that states if the term was contained in the title or in the abstract. All terms are stemmed, lower cased and words that just contain punctuation are stripped.
While iterating through the documents also the patent_info file is created.

The next step is to consolidate the sorted list of words and generate dictionary and postings file. Same words are summed up and their occurences are united by iterating through the word list. For each term one entry is written down to the dictionary and the postings file according to the specified form below (see generated documents).

At the same time a document length vector is created that states patentNo and normalization factor ( sqrt(pow(tf,2)) ) for each document. Finally the vector is written down to the last line of the postings file (see generated documents) to be accessable during the search where a vector space model including length normalization will be applied.


== Architecture SEARCHING ==

files: search.py
generated files: output.txt

Regarding the search we tried several different approaches to come up with our current / final result. In the following we want to point out our final structure as well as our thoughts, experiments and outcomes:


~ Final result:

Preparation that is needed for the search process contains 
- reading in and processing the query
		The query xml-file is parsed with the support of "ElementTree". Words in titel and description are tokenized, lower cased and stemmed in the same way as the patents before and added to a list of terms. They are treated as equally as manual research has shown that the used words used in the title of the query follow rather descriptions of the usage and are mostly not more specific than in the description. (for further aspects see below in "NOSIMMERGE") 
- reading dictionary as well as patent info into memory 
		data format in both cases a dictionary
- getting length vector for normalization
		(last line in postings file)
- getting line positions to access postings file
		Line positions is a list of offsets (positions) that contains the starting position of each line in a file. As our dictionary points to lines of the postings file we needed a way to access line content without reading the whole file into memory. We deal with this task by using the line positions and the function seek(). 

We apply the Vector Space Model (implemented in an external class) that generates a list of documents (patentNo) and their scores ordered by decreasing score. (additional information in secation below)

Furthermore...

All results are written down to the output file.


~ Thoughts, experiments and outcomes:

We started with a vector space model implemented in VectorSpaceModel.py as a base. Furthermore the goal was to add additional methods to improve our outcomes. The recall was always good as all relevant documents could be retrieved. Some of them are not ranked high, though. We tried to improve the ranking of the VSM scores with reveral methods. In the end we generated the best outcome by using our implementation of the VSM including Stopwords.
The algorithms are described and discussed in the following regarding their application reasons and outcomes.

Vector Space Model:
	APPLIED 

	Implementation:
	After processing the query (lower case, stemming, strip word when just punctuation) the cosine score algorithm is applied: 
	Query and document are represented as weighted tf.idf vectors (no idf for documents). Afterwards normalization is applied and the cosine similarity is computed. The documents are ordered by their resulting scores.

	Stopwords - are removed! Contained are those defined by nltk.corpus.stopwords.words('english'), some manually added words
		(It was also experimented with adding words to the stopwords that seem to occur frequently without a lot of meaning in the context - the results were diverse but not better reguarding the given test queries - gave a boost for the relevant documents that were already ranked well, some others decreased in the ranking - finally we applied these stopwords, too)


SIMNOMERGE: measures similarity between query and patent xml trees
	NOT APPLIED 
	- Reason: Especially title in query and patent did not seem to be on the same language level regarding the choice of words and structure of the title. In the query, the title describes the usage of the searched patent while the title in the patent describes rather the method. The words used in the query are also rather basic, not so specific and rather structured like a sentence compared to the normalization style in the patent. As those tags do not seem to be on the same language and content level we decided not to use this algorithm.

Pseudo Relevance Feedback:
	APPLIED

	Implementation:
	take n best results of scores produced by VSM. Add (read in, tokenized and processed) terms of the best resulting patents and add them to the existing query. Just the highest ranked k terms regarding the tf.idf in the documents are taken.
	Afterwards the query is processed again. The values of n and k was decided from doing manual experiments with the two queries that we had got and the final value that we used was n=10 and k=10.

	Improvement:
	To make the original query words be more important in the search than the new ones, we let the new words get the tf value 0.5 and the original ones keeps their real tf value. However, if an old term appeared in the new query list, the tf is set to 1.5 times the original value. This idea is taken from the paper Larson, R. Ray 2010, Blind Relevance Feedback for the ImageCLEF Wikipedia Retrieval Task, http://ceur-ws.org/Vol-1176/CLEF2010wn-ImageCLEF-Larson2010.pdf

Field / zone treatment:
	see IPC

IPC: A classification system for patents
	TRIED
	- Outcome:
	not applied because doesn't deliver a lot of value as mostly the same IPC group is found anyway
	most of the docs that we have to retrieve have the same IPC

	Implementation:
	1. Do a first unconstrained search, see which IPC seems to be most relevant, and then search only for documents that have that IPC
	2. Guess the IPC from the query terms

Query Expansion: using synonyms from Word Net (https://wordnet.princeton.edu/) to expand the query
	- Reason: From looking at the synonyms given by WordNet we were under the impression that our problem with relevant documents that are retrieved badly, would not be solved, since they were not containing these synonyms either. So this method doesn't seem like a good solution to retrieve those documents and we decided instead to focus on Pseudo Relevance Feedback.

Positional Index and Phrasal Queries
	APPLIED

Latent Semantic Analysis:
	NOT APPLIED
	- Reason: time, seems to be promising in patent information retrieval in general and regarding our results: 
	Our difficulty was always that we could find all documents but some of them were retrieved rather in the end. Manual comparision could show that the main reason for this might be that some of the relevant documents hardly have any overlapping words, the content seems to be fitting, though. So actually what we really want to do is compare the meanings or concepts behind the words. LSA attempts to solve this problem by mapping both words and documents into a "concept" space and doing the comparison in this space.

Run-time optimization:
	The test queries are solved easily within two minutes, so no further adjustments are made. 

Further thoughts / ideas how to improve results:
- Modify scores according to # of citations, publication year
	° Rank highly cited documents higher
	° Rank documents by well known writers higher
	° recently created patents are more important than older ones
	--> distribution of the publication years gives the impression that recent years could be a little more important in query 1, query 2 makes us reject this hypothesis, though
		Q1	
		Relevant	Irrelevant
		1994 1		1979 1
		1995 1		1988 1
		1996 1		1989 1
		1997 1		1990 1
		2001 1		1991 1
		2005 1		1992 1
		2007 1		1997 1
		2008 5		2000 2
		2009 1		2002 2
		2010 3		2003 1
		2011 5		2005 1
		2012 1		2006 1
		 			2007 2
		 			2008 1
		 			2010 1
		 			2011 4
		 			2012 3
			
		Q2	
		Relevant	Irrelevant
		2008 4		2008 7
		2009 7		2009 4
		2010 9		2010 7
		2011 5		2011 7

- expand resulting patents by looking at
	° What the author / invetor has done
	° Other documents with same classification (IPC)
	° Other documents that has cited highly ranked results
	° Other references
- Relevance feedback from IPC, the inventor, cited documents
- Complement to stopwords: 
	Have threshold for low idf and remove those words to not get a lot of unrelevant documents


---


The main files that we used are built in the following structure:

dictionary_file:	contains a dictionary that is written down in a file 
                    form: "term frequency line-in-postings-file" 
					the line-in-postings-file is meant like a pointer to the specific line in the postings_file
postings_file:  	contains all postings belonging to the different dictionary entries
                    form: "patentNo    (1+ log_tf)   isTitle   isAbstract" 
                    (isTitle and isAbstract are binary boolean values --> 1 or 0)
                    LAST LINE: contains information about the length vector / normalization factor per patent
                    form: "patentNo   normalizationFactor"
output_file:    	shows all file numbers that result from the queries
                    form: "file_name <space> file_name <space>..."
patent_info:		contains information about the different patents
					form: "patentNo |  Publication Year  |  # Cited by |  IPC Primary  | 1st Inventor"


== Files included with this submission ==

PYTHON FILES
index.py        			Used for creating dictionary and postings list
search.py       			Reads the dictionary file into memory, formats and solves queries with the information in the postings file
VectorSpaceModel.py 		class containing implementation for applying VSM
PseudoRelevanceFeedback.py  class containing implementation for applying PRF
IPC.py 						class containing implementation for using the IPC
text_processing.py 			Includes general functionality for applying case-folding, stemming and stopping

TEXT FILES
dictionary.txt  The dictionary
postings.txt    The postings list
patent_info.txt information extracted from patent corpus structured by patent ID

README.txt      Information about the submission (this file)


== Allocation of work ==

discussion and conception  			A0135280M, A0135622M, A0134784X
basic indexing 						A0135280M, A0135622M  
basic searching 					A0135280M, A0135622M
debug prints of results				A0134784X
vector space model  				A0135280M - main parts of the code produced in precvious assignment by A0135280M, A0135622M
pseudo relevance feedback 			A0135280M
positional index/phrasal queries 	A0134784X 
IPC 								A0134784X (deep research and implementation)
Latent Semantic Analysis			A0135622M (research)
README.txt 							A0135622M


== Statement of individual work ==

Please initial one of the following statements.

[X] We, A0135280M, A0135622M and A0134784X certify that we have followed the CS3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] We, A0135280M, A0135622M and A0134784X did not follow the class rules 
regarding homework assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==
* We used the cosine score algorithm that was teached in the lesson.
* We used a solution proposed on this page to get the correct line position used for the seek() method: 
    http://stackoverflow.com/questions/620367/python-how-to-jump-to-a-particular-line-in-a-huge-text-file 
* https://docs.python.org/2/reference/datamodel.html

