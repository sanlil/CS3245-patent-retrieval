A0135280M and A0135622M
mail: a0135280@u.nus.edu and a0135622@u.nus.edu

== General Notes about this assignment ==

Architecture:
We used the two main files index.py and search.py to index the given files and process the given free text queries.
In both files we generated the main functions indexing(...) and search(...) to execute those tasks.
Furthermore we created several additional functions to abstract the code and make it better readable.
More precise explanations can be found in the function comments in the code.

The main files that we used are built in the following structure:
dictionary_file:	contains a dictionary that is written down in a file 
                    form: "term frequency line-in-postings-file" 
						the line-in-postings-file is meant like a pointer to the specific line in the postings_file
postings_file:  	contains all postings belonging to the different dictionary entries
                    form: "file_name <space> file_name <space>..."
                    the pre-last line contains the names of all given documents in the training data
                    the last line contains the length vector (all normalization factors from 0 until the highest document ID)
output_file:    	shows all file numbers that result from the queries
                    form: "file_name <space> file_name <space>..."

Experiments:
	We tried an indexing version with and without digits in our dictionary terms (as you can see in the essay questions) 
	but decided to use the full version (with digits) since the searches then can be more precise.

    We also used time.time() to evaluate if changes did the code more efficient or not.

== Files included with this submission ==

index.py        Used for creating dictionary and postings list
search.py       Reads the dictionary file into memory, formats and solves queries with the information in the postings file
dictionary.txt  The dictionary
postings.txt    The postings list
ESSAY.txt       Answers to the essay questions
README.txt      Information about the submission

== Statement of individual work ==

Please initial one of the following statements.

[X] We, A0135280M and A0135622M, certify that we have followed the CS3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] We, A0135280M and A0135622M, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

I suggest that I should be graded as follows:

<Please fill in>

== References ==
* We used the cosine score algorithm that was teached in the lesson.
* We used a solution proposed on this page to get the correct line position used for the seek() method: 
    http://stackoverflow.com/questions/620367/python-how-to-jump-to-a-particular-line-in-a-huge-text-file 
* We used a solution proposed on this page as an inspiration to get indices of N maximum values in a numpy array:
	http://stackoverflow.com/a/6910672

