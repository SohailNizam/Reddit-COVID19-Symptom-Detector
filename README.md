# Reddit COVID19 Symptom Detector
All files associated with Bio-NLP Assignment 1 (COVID Symptom Surveillance) 

The code for this project is in Annotator_System_Code.py.

The results from my annotation of the UnlabeledSet2.xlsx file are in the file UnlabeledResults.xlsx. This file is ready to be passed to the evaluation script.

To run the code, make sure all files (aside from GS_Results and Unlabeled_Results) are saved in the same directory. If saved in current working directory, code can be run immediately. If not, open the code and change the path variable at the top to the location of the files. Currently set to annotate the Gold Standard comment set. If you wish to annotate a different xlsx file, change the 'infilename' variable to the name of the desired file, and the 'outfilename' variable to the desired output file name. 

This code takes in a .xlsx file containing a column of social media posts (Reddit comments) and a column of unique identifiers, and it returns a .xlsx file containing those same 2 columns plus a column of '$$$' seperated Concept Unique Identifiers (defined in the Twitter Symptom Lexicon) as well as a columns of '$$$' seperated negation flags (1 if symptom is negated, 0 if not). 
