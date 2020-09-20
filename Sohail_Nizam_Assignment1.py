# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:30:25 2020

@author: Sohail Nizam
"""

import string
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import itertools
import Levenshtein
import numpy as np
from os import listdir
import math


###################### FOR THE USER ######################
'''
To run this code, choose a path in which all of the following files are stored:
-The file to be annotated (eg Assignment1GoldStandardSet.xlsx)
-COVID-Twitter-Symptom-Lexicon.txt
-neg_trigs.txt
-Nizam_Annotations(s13).xlsx

Take this path and set the path variable below equal to it.
Set the variable infilename equal to the file you wish to annotate.
Set the variable outfilename equal to the name you'd like to give to the
output file.
Then just run the whole code. The excel file with the annotations will appear in 
the location set by the path variable.
'''

path = './'
infilename = 'Assignment1GoldStandardSet.xlsx'
outfilename = 'GS_Results.xlsx'

###################### Import all files ###################### 

# Import original symptom lexicon, store in dict
lex = open(path + 'COVID-Twitter-Symptom-Lexicon.txt')
#Store symptoms in dictionary
symptom_dict = {}
for line in lex:
	#print(line)
	items = line.split('\t')
	symptom_dict[str.strip(items[-1])] = items[1]


# Import original negation list
neg_infile = open(path + 'neg_trigs.txt')
negations = []
for line in neg_infile:
	#print(line)
	negations.append(str.strip(line))

# Add some one negation based on my annotations
negations += ["never"]
	

# Create second symptom dict out of my annotations
annotations = pd.read_excel(path + 'Nizam_Annotations(s13).xlsx')
new_symptom_dict = {}
for index, row in annotations.iterrows():
	symptoms = str(row['Symptom Expressions']).split('$$$')[1:-1]
	cuis = str(row['Symptom CUIs']).split('$$$')[1:-1]
	
	for symptom,cui in zip(symptoms, cuis):
		
		#don't add symptoms with negations
		for neg in negations:
			if neg not in symptom and symptom != 'nan':
				new_symptom_dict[symptom] = cui

# Add my symptom dict to the original
for symptom,cui in new_symptom_dict.items():
	if symptom not in symptom_dict.keys():
		symptom_dict[symptom] = cui
				

###################### Build the System ######################

# window function from lecture
def sliding_window(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
	#creates an object which can be iterated one element at a time
    word_iterator = iter(words)
	#islice() makes iterator that returns selected elements from word_iterator
    word_window = tuple(itertools.islice(word_iterator, window_size))
    yield word_window
    #now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window


				
def is_negated(symptom, sentence, negations):
	'''
	Is the first word of the symptom within 3 words of any negation
	with no periods or other negations in between?
	'''

	#get index of beginning of first word of symptom
	symptom_ind = re.search(re.escape(symptom), sentence).start()
	
	#truncate the string just before the symptom
	sentence_trunc = sentence[:symptom_ind]
	#print(sentence_trunc)
	#print()
	
	#tokenize the truncted sentence
	#and take just the final 3 words
	word_list = list(nltk.word_tokenize(sentence_trunc))
	word_list = word_list[-3:]
	#print(word_list)
	#print()
	
	#search these final three words for a negation
	symptom_negated = False
	for neg in negations:
		if re.search(r'\b' + neg + r'\b', ' '.join(word_list)):
			symptom_negated = True
			#print(symptom + ' is negated by ' + neg)
		
			break
	
	return(symptom_negated)


def is_used(word_list, start, used_dict):
	'''
	A small helper function to check if a particular
	window of words contains any words that have already
	been counted as part of another symptom.
	'''
	word_tup_list = []
	for word in word_list:
		word_tup_list.append((word, start))
		start += 1
	
	any_used = False
	for word_tup in word_tup_list:
		if used_dict[word_tup] == 1:
			any_used = True
			break
	
	return(any_used)

def is_match(window_string, symptom_dict, thresh):
	'''
	This function takes a string containing several words,
	a symptom dictionary, and a threshold (float).
	
	It takes the string, checks the levenshtein ratio with 
	every symptom in sympmtom_dict, finds the highest ratio,
	and if that ratio is above thresh, it returns a tuple containing
	True and the corresponding matching symptom: (True, symptom)
	If the closest match does not cross the threshold, the boolean is False:
	(False, symptom)
	'''
	
	best_match = ''
	best_similarity = -1
	for symptom in symptom_dict.keys():
		similarity_score = Levenshtein.ratio(window_string, symptom)
		if similarity_score > best_similarity:
			best_match = symptom
			best_similarity = similarity_score
	
	
	#revert the match to False if 'no' appears in the window
	#That way we don't count "no fever" as fever
	#and we don't count "fever no" when the sent is "no fever no chills"
	#Else, chills' negation would be lost for no reason
	#we'll catch the symptom with a smaller window
	match_bool = False
	if best_similarity > thresh and re.search(r'\bno\b', window_string) == None:
		match_bool = True
		
	
	match_tup = (match_bool, best_match)
		

	return(match_tup)
	

def get_symptoms(sent, symptom_dict, negations, thresh):
	
	'''
	This function takes a single sentence, a symptom dict
	a list of negations, and a fuzzy matching Levenshtein ratio threshold.
	
	It returns a list of 4 lists. First list contains all found non-standard
	symptom expressions. Second list contains all standard symptom expressions
	(as written in the inputed dictionary). Third list contains all CUIs. Fourth
	list contains all negation flags.
	'''
	
	#Do this to take care of inconsistencies created by
	#tokenizing contractions like  "don't" which results in "do n't"
	sent = ' '.join(list(nltk.word_tokenize(sent)))
	sent = re.sub(r"\bn't\b", "not", sent)
	
	words = list(nltk.word_tokenize(sent))
	
	#this dict keeps track of whether each word has been counted in a symptom
	used_dict = {}
	for index, word in enumerate(words):
		used_dict[(word, index)] = 0
		
	
	symptoms = [[], [], [], []]
	#use every possible symptom length as a window size
	#But start with the largest window and decrease. Why?
	#Because if we hit 'Sore Throat', we want to scratch out that symptom
	#So we don't get 'Sore' again
	for i in range(len(words), 0, -1):
		
		#initialize window position variable
		window_start = 0
		
		#slide the window, check for symptom at each stop
		for window in sliding_window(words, i):
			
			#print(window)
			 
		    #get a string rep of this window to
			#check against the symptom_dict
			window_string = ' '.join(window)
			
			#get a list rep of this window to
			#check against the used_dict
			window_list = list(window)
			
			#store a boolean value
			#True if any of the words have already been counted as part of
			#a symptom.
			#False if none of them have.
			any_used = is_used(window_list, window_start, used_dict)
			
			#print(any_used)
			
			#store a tuple of boolean value and the closest matching string
			#the boolean is true if the closest match crosses the set
			#threshold
			match_tup = is_match(window_string, symptom_dict, thresh)
			#print(match_tup)
			#print()
			match = match_tup[0]
			matching_symptom = match_tup[1]
			
		
			if match and not any_used:
				
				   
				#if it's a fresh symptom, check if it's negated
				if is_negated(window_string, sent, negations):
					#print(sent + '\t' + symptom_dict[window_string] + '-neg')
					#print('---')
					#print()
					neg = str(1)
					
				else:
					#print(sent + '\t' + symptom_dict[window_string] + '\t')
					#print()
					#print('---')
					neg = str(0)
				
				#Add the symptom, the cui, and the negation flag
				#to their respective sublists in the output   
				symptoms[0].append(matching_symptom)
				symptoms[1].append(matching_symptom)
				symptoms[2].append(symptom_dict[matching_symptom])
				symptoms[3].append(neg)
				
				#change each of the words in this window to 'used'
				#status in used_dict
				# TODO
				position = window_start
				for word in window_list:
					used_dict[(word, position)] = 1
					position += 1
				
			
			#update the window starting position
			window_start += 1

	return(symptoms)




def symptom_detector(in_df, symptom_dict, negations, thresh):
	
	'''
	This function takes in a pandas dataframe consisting of 7 columns
	(reddit username, date, comment, symptom expressions, standard symptom,
	symptom CUI, and negation flag) as well as a dictionary with non-standard 
	symptom expression keys and CUI values, and a list of negations to consider.
	
	The symptom expressions, standard symptoms, and negation columns may or may
	not be empty (depending on whether it's a new set or an evaluation set).
	
	This function goes row by row in the dataframe, evaluating each comment,
	searching for symptoms, checking for their negations, and filling in the 
	empty columns of a copy of the original df.
	
	The output is a new dataframe with all columns filled in.
	'''
	
	
	# Make a new data frame with the ID and TEXT copied
	# and the remaining columns empty
	df = in_df.loc[:, ['ID','TEXT']]
	df['Symptom Expressions'] = ''
	df['Standard Symptom'] = ''
	df['Symptom CUIs'] = ''
	df['Negation Flag'] = ''
	
	
	# for each row in the df
	for i, row in df.iterrows():
		print(i)
	
		#get just the comment
		#do the extra str() just in case
		#there's something wonky in the comment
		comment = str(row['TEXT'])
		
		#tokenize the comment into sentences
		sentences = sent_tokenize(comment)
		
		# Go sentence by sentence looking for symptoms
		# and their possible negations
		# append the info to the following lists
		expression_list = []
		standard_list = []
		cui_list = []
		neg_list = []
		
		for sent in sentences:
			#get_symptoms provides alist of lists
			#one list each for expressions, standard symtoms, cuis, neg flags
			all_info = get_symptoms(sent, symptom_dict, negations, thresh)
			expression_list += all_info[0]
			standard_list += all_info[1]
			cui_list += all_info[2]
			neg_list += all_info[3]
		
		#a running dict of cuis with negation flags
		cui_dict = {}
		for cui, neg in zip(cui_list, neg_list):
			
			
			#if we haven't added the cui at all, add it
			if cui not in cui_dict.keys():
				cui_dict[cui] = neg
			
			#if we've added it before and now have a dispute
			#(ie current neg flag diff from past neg flag)
			#just change the flag to 0. No matter what,
			#if there's a dispute, it means we saw at least
			#one neg = 0. That's enough for a positive overall flag.
			elif cui in cui_dict.keys() and cui_dict[cui] != neg:
				cui_dict[cui] = str(0)
				
			
			
			
	    
		cui_list_clean = list(cui_dict.keys())
		neg_list_clean = list(cui_dict.values())
		# Take those 4 lists and add their info to the correct columns
		# as $$$ separated strings, with $$$ at beginning and end
		#df.at[i, 'Symptom Expressions'] = '$$$' + '$$$'.join(expression_list) + '$$$'
		#df.at[i, 'Standard Symptom'] = '$$$' + '$$$'.join(standard_list) + '$$$'
		df.at[i, 'Symptom CUIs'] = '$$$' + '$$$'.join(cui_list_clean) + '$$$'
		df.at[i, 'Negation Flag'] = '$$$' + '$$$'.join(neg_list_clean) + '$$$'
	
	   


	#return the modified df with the symptom info entered
	return(df)


# Use the system on chosen datset #
comment_df = pd.read_excel(path + infilename)
labeled_df = symptom_detector(comment_df, symptom_dict, negations, .90)
just_labels = labeled_df.drop('TEXT', axis = 1)
just_labels.to_excel(path + outfilename)








    