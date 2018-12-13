import os
import json
import fastText
import numpy as np
from sklearn.cluster import KMeans  
from sklearn.manifold import TSNE
import random
import typing 
import pandas as pd
import re
from prettytable import PrettyTable
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import os
import progressbar
import slate
import nltk
nltk.download('wordnet')

def printTable():
	"""
	Show System Instruction
	"""
	intro = PrettyTable(['Table Extraction v0 (Author: Shixiang Zhu, THOR Group)'])

	intro_list = []
	intro_list.append("This system will give  you related table based on the table you entered")
	intro.add_row(intro_list)

# 	intro_list = []
# 	intro_list.append("Step 1: You need to specify a topic that related to disaster. (e.g The hurricane harvey in Huston) \n Step 2: Based on the topic you provided, the system will suggest some keywords and hashtags to you. \n Step 3: You could either choose keywords and hashtags from the provided lists or from your own words.")
# 	intro.add_row(intro_list)
	print(intro)
	print(" ")

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Can be used in loop to create prograss bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()


def read_json1(directory:str):
	"""
		input:
		directory: str
			the directory that  stores the  original data
		return:
			dict_paper_table: typing.DIct
				paper's index, paper name -> list of table caption
			dict_list_index: typing.Dict
				paper's index -> paper name
		"""
	dict_paper_table = {}
	index_paper = 1
	dict_list_index  = {}
	for filename in os.listdir(directory):
		dict_list_index[ "p" + str(index_paper)] = filename
		key = "" + str(index_paper) + ", " + filename
		with open(directory + "/" + filename, 'r') as f:
			list_table = json.load(f)
			list_caption = []
			for data in list_table:
				if data["figType"] == "Table":
					list_caption.append(data["caption"])
				if(len(list_caption) > 0):
					dict_paper_table[key] = list_caption
				index_paper += 1
	return dict_paper_table, dict_list_index

def read_json(directory:str):
	"""
		input:
		directory: str
			the directory that  stores the  original data
		return:
		df: pd.Dataframe
			paper_name paper_ind table_ind table_cap
		"""
	df = pd.DataFrame(columns = ['paper_name', 'paper_ind', 'table_ind', 'table_cap'])
	index_paper = 0
	for filename in os.listdir(directory):
		with open(directory + "/" + filename, 'r') as f:
			list_table = json.load(f)
			list_caption = []
			table_index = 0
			for data in list_table:
				if data["figType"] == "Table":
					dfTem = pd.DataFrame([[filename, index_paper, table_index, data["caption"]]], columns = ['paper_name', 'paper_ind', 'table_ind', 'table_cap'])
					df = df.append(dfTem, ignore_index=True)
					table_index += 1
		index_paper += 1
	return df

def get_sentence_vector_simlarity_fasttext(caption, user_input):
	"""
		input:
		caption: nparray
			table caption
		user_input: str
			user input
		return:
		   list of tuples sorted by similarity with user's  input
		   
		"""
	dict_index_sim = {}
	ftmodel = fastText.load_model('wiki.en.bin')
	user_input = sen_clean(user_input)
	vec_list = []
	for sen in caption:
		vec_list.append(ftmodel.get_sentence_vector(sen_clean(sen).strip()))
	user_input = ftmodel.get_sentence_vector(sen_clean(user_input).strip())
#     print(user_input.__class__)
	for i in range(len(vec_list)):
		table = np.asarray(vec_list[i])
		sim = similarity(table, user_input)
		dict_index_sim[i] = sim
	sorted_by_value = sorted(dict_index_sim.items(), key=lambda kv: -1 * kv[1])
	return sorted_by_value
   




def get_sentence_vector_simlarity_tfidf(caption, user_input):
	"""
		input:
		caption: nparray
			table caption
		user_input: str
			user input
		return:
		   list of tuples sorted by similarity with user's input
		"""
	dict_index_sim = {}
	user_input = sen_clean(user_input.strip())
	vec_list = []
	vec_list.append(sen_clean(user_input).strip())
	for sen in caption:
		vec_list.append(sen_clean(sen).strip())
	senToVec = TfidfVectorizer(lowercase=False, norm = 'l2') 
	tfidfRes = senToVec.fit_transform(vec_list)
	usetVec = tfidfRes[0]
	senVec = tfidfRes[1: tfidfRes.shape[1]]
	res = usetVec.dot(senVec.T).toarray()
	for item in res:
		for i in range(0, len(item)):
			dict_index_sim[i] = item[i]
	sorted_by_value = sorted(dict_index_sim.items(), key=lambda kv: -1 * kv[1])
	return sorted_by_value, dict_index_sim

def similarity(v1, v2):
	n1 = np.linalg.norm(v1)
	n2 = np.linalg.norm(v2)
	return np.dot(v1, v2) / n1 / n2



def sen_clean(sen: str) -> str:
	"""
		This method is used for clean text data
		Args:
			sen: str
				tweet text
		Returns:
			sen: str
				cleaned text
		"""
	sen = sen.lower()
	sen = re.sub('[^a-zA-Z0-9!]+', ' ', sen) # remove not valid characters
	lemmatizer = WordNetLemmatizer()
	wordList = sen.split(" ")
	
	res = ""
	for word in wordList:
		if "\n" in word:
			continue
		res = res + lemmatizer.lemmatize(word) + " "
#     print(res.strip())
	return res



def print_search_engine_result(sorted_by_value: typing.List, df):
	"""
	This  method  is used for printing search engine result
	Args:
		sorted_by_value: typing.List
			sorted result
		df: DataFrame
			paper
	Returns:
	"""
	top_10 = []
	for item in sorted_by_value:
		print(df.get_value(item[0], 'table_cap'))
		top_10.append(df.get_value(item[0], 'table_cap'))
	print(" ")
	for item in sorted_by_value:
		print(df.get_value(item[0], 'paper_name'))
	return top_10


def readPaper(path):
	"""
	This  method is used for reading the paper in
	Args:
	Returns:
		list of paper name
		list of paper content
	"""
	print("In the process of reading papers")
	
	dict_paper_name_text = {}
	paperName = []
	paperContent = []
	count = 0
	for filename in os.listdir(path):
		count += 1
	# printProgressBar(0, count, prefix = 'Progress:', suffix = 'Complete', length = 50)
	count2 = 0
	for filename in os.listdir(path):
		# printProgressBar(count2 + 1, count, prefix = 'Progress:', suffix = 'Complete', length = 50)
		count2 += 1
		if ".pdf" not in filename: continue

		with open(path + filename, "rb") as f:
			pdf = slate.PDF(f)
			os.system('cls||clear')
			res = ""
			for page in pdf:
				res = res + sen_clean(page).strip()
				# print(res)
			dict_paper_name_text[filename] = res
			paperName.append(filename)
			paperContent.append(res)
	return paperName, paperContent

def baseline_methods(user_input, paperName, paperContent, root_dir):
	
	df = read_json(root_dir)
	# print (df)
	# user_input = input("Please enter table caption: ")
	print("##################################################################################################################")
	print("Method 1: wiki_pre-trained_fasttext baseline")
	
	# fasttext 
	sorted_by_value_cap_ft = get_sentence_vector_simlarity_fasttext(df["table_cap"].values, user_input)
	# print(sorted_by_value_cap_ft)
	# print_search_engine_result(sorted_by_value, df)
	dict_index_sim_cap_ft = {} # Stores the rank of caption (papername, rank)
	countCap_ft = 0
	for tup in sorted_by_value_cap_ft:
		dict_index_sim_cap_ft[tup[0]] = countCap_ft
		countCap_ft += 1
	# print(dict_index_sim_cap_ft)
	new_cap_rank_column_ft = [] # caption rank based on the index  of df (ft)
	for i, row in df.iterrows():
		new_cap_rank_column_ft.append(dict_index_sim_cap_ft[i])
	# print(new_cap_rank_column_ft)
	print_search_engine_result(sorted_by_value_cap_ft[0:10], df)

	# tfidf
	print("##################################################################################################################")
	print("Method 2: tfidf baseline")
	sorted_by_value_cap, dict_index_sim_cap = get_sentence_vector_simlarity_tfidf(df["table_cap"].values, user_input)
	# print(sorted_by_value_cap)
	dict_index_sim_cap = {} # Stores the rank of caption (papername, rank)
	countCap = 0
	for tup in sorted_by_value_cap:
		dict_index_sim_cap[tup[0]] = countCap
		countCap += 1
	# print(dict_index_sim_cap)
	new_cap_rank_column = [] # caption rank based on the index  of df
	for i, row in df.iterrows():
		new_cap_rank_column.append(dict_index_sim_cap[i])
	# print(new_cap_rank_column)

	print_search_engine_result(sorted_by_value_cap[0:10], df)

	#tfidf paper + tfidf caption average ranking
	print("##################################################################################################################")
	print("Method 3: tfidf paper + tfidf caption average ranking")
	# paperName, paperContent = readPaper()
	sorted_by_value, dict_index_sim_paper = get_sentence_vector_simlarity_tfidf(paperContent, user_input)

	dict_index_sim_paper = {} # Stores the rank of paper (papername, rank)
	countPaper = 0
	for tup in sorted_by_value:
		dict_index_sim_paper[paperName[tup[0]][:-4] + ".json"] = countPaper
		countPaper += 1
	# print(dict_index_sim_paper)

	new_paper_rank_column = [] # paper rank based on the index  of df
	weighted_rank = []
	for name in df["paper_name"].values:
		# print(name)
		name = name
		new_paper_rank_column.append(dict_index_sim_paper[name])

	# print(len(new_paper_rank_column), len(new_cap_rank_column))

	for i in range(0, len(new_paper_rank_column)):
		weighted_rank.append((new_paper_rank_column[i] + new_cap_rank_column[i]) / 2.0)
	# print(weighted_rank)
	df['weighted_rank'] = weighted_rank
	df['cap_rank_tfidf'] = new_cap_rank_column
	df['paper_rank'] = new_paper_rank_column
	df = df.sort_values(by=['weighted_rank'])
	cap_print = df["table_cap"].values.tolist()
	cap_paper = df["paper_name"].values.tolist()
	cap_print = cap_print[0:10]
	cap_paper = cap_paper[0:10]
	for i in cap_print:
		print(i)
	print(" ")
	for i in cap_paper:
		print(i)


def main():
	printTable()
	user_input = input("enter")
	path  = "/Users/zhengshuangjing/Desktop/tableDatabase/Entity_Rsurvey/"
	paperName, paperContent = readPaper(path)
	root_dir =  "/Users/zhengshuangjing/desktop/tableDatabase/mergedJson"
	baseline_methods(user_input, paperName, paperContent, root_dir)
	
	
	# df['weighted_rank'] = weighted_rank
	# df['cap_rank_tfidf'] = new_cap_rank_column
	# # df['cap_rank_ft'] = new_cap_rank_column_ft
	# df['paper_rank'] = new_paper_rank_column
	# df = df.sort_values(by=['weighted_rank'])
	# print(df)
	# print(df["table_cap"].values)




# 	'''
# 	Calculate paper rank
# 	paperName  [] list
# 	paperList [] list
# 	get tfidf of paperlist [] and query

# 	calculate simList []
# 	paperName simList ==> dict (paperName,  simList) ==>  change  value to dict (paperName, paperRank)  ==> append into dataframe
# 	'''

# 	'''
# 	Calculate Caption Rank:
# 	append into  datagrame
# 	'''
if __name__ == '__main__':
	main()




	   









