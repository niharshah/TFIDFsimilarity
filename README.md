# TFIDFsimilarity
Computes a similarity matrix between reviewers and papers using TF-IDF.

How to use:
- You may have to install the nltk library (pip install nltk)
- Create the following two folders:
    - "Paper_Folder": Contains text files where each text file contains the text of a paper
    - "Reviewer_Folder": Contains text files where each text file contains the text of a potential reviewer's past papers
- Call compute_similarities(Reviewer_Folder, Paper_Folder) 
- The input to this function are the addresses of the two folders
- The output is the list of reviewers indexed by their respective text filenames, list of papers indexed by their respective text filenams, and the similarity matrix
- The similarity matrix has one row per reviewer and one column per paper, where the rows and columns are ordered according to the aforementioned lists

ACKNOWLEDGMENTS:
- This code implements "The Toronto Paper Matching System: An automated paper-reviewer
assignment system," Charlin and Zemel, 2013  (which is not open source)
- Initial version was written by Han Zhao as part of the paper: "On strategyproof conference peer review," Xu, Zhao, Shi, and Shah, 2019. 
- Please acknowledge the aforementioned papers if you use this code.
- Code subsequently modified by Nihar Shah to enable for a more general purpose use
