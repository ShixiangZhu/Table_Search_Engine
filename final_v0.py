import functionBank as fb
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
import nltk

nltk.download('wordnet')

def read_json(directory:str):
	"""
		input:e
		directory: str
			the directory that  stores the  original data
		return:
		df: pd.Dataframe
			paper_name paper_ind table_ind table_cap
		"""
	maxPaper = 0
	maxTable = 0
	df = pd.DataFrame(columns = ['paper_name', 'paper_id', 'table_id', 'table_cap'])
	index_paper = 0
	table_index = 0
	for filename in os.listdir(directory):
		# if(index_paper > 3): break
		temp = table_index
		with open(directory + "/" + filename, 'r') as f:
			list_table = json.load(f)
			list_caption = []
			for data in list_table:
				if data["figType"] == "Table":
					table_id = "t" + str(table_index)
					paper_id = "p" + str(index_paper)
					# print(paper_id)
					dfTem = pd.DataFrame([[filename[0:-5], paper_id, table_id, data["caption"]]], columns = ['paper_name', 'paper_id', 'table_id', 'table_cap'])
					df = df.append(dfTem, ignore_index=True)
					table_index += 1
		index_paper += 1
		if table_index == temp:
			index_paper -= 1
	maxPaper = index_paper
	maxTable = table_index
	print(maxPaper, maxTable)
	return df, maxPaper, maxTable

def getWordSet(list_corpus) -> typing.Dict:
	"""
	input: list_corpus -> typing.List
	return: dict => key = word, value = ind
	"""
	dict_word = {} # key: word,  value: count
	for sen in list_corpus:
		sen = fb.sen_clean(sen)
		senList = sen.split(" ")
		for word in senList:
			if word in dict_word:
				count = dict_word[word]
				dict_word[word] = count + 1
			else:
				dict_word[word] = 1
	word_corpus = {} # key = word, value = ind
	word_ind = []
	wordInd = 1
	corpus_word = {} # key = ind (w1), value = word
	for key, value in dict_word.items():
		word_corpus[key] = "w" + str(wordInd)
		corpus_word["w" + str(wordInd)] = key
		wordInd += 1

	# print(word_corpus)
	return word_corpus, corpus_word

def getMostSim(list_sen, list_ind):
	"""
		input:
		list_sen: list
		list_ind: list
		return:
		   list of most similar id
		"""
	list_sim_ind = []
	vec_list = []
	for sen in list_sen:
		vec_list.append(fb.sen_clean(sen).strip())
	senToVec = TfidfVectorizer(lowercase=False, norm = 'l2') 
	tfidfRes = senToVec.fit_transform(vec_list)
	for i in range(tfidfRes.shape[0]):
		temp = tfidfRes[i].dot(tfidfRes.T).toarray()
		for item in temp:
			maxVal = 0
			index = 0
			for j in range(0, len(item)):
				if j != i and list_ind[i] != list_ind[j]:
					if maxVal < item[j]:
						index = j
						maxVal = item[j]
			# print(maxVal)
			list_sim_ind.append(list_ind[index])
	return list_sim_ind

def stringToInt(inStr, maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word) -> str:
	if inStr[0] == "p":
		dict_ind_paper[str(int(inStr[1:]))] = inStr
		return str(int(inStr[1:]))
	if inStr[0] == "t":
		dict_ind_table[str(int(inStr[1:]) + maxPaper)] = inStr
		return str(int(inStr[1:]) + maxPaper)
	else:
		dict_ind_word[str(int(inStr[1:]) + maxTable + maxPaper)] = inStr
		return str(int(inStr[1:]) + maxTable + maxPaper)

def writeFile(dict_ind_table, dict_ind_paper, dict_ind_word, maxPaper, maxTable, df):
	file = open("paper.edgelist", "w")
	for index, row in df.iterrows():
		file.write(stringToInt(row['paper_id'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
		file.write(" ")
		file.write(stringToInt(row['table_id'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
		file.write('\n')

		file.write(stringToInt(row['table_id'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
		file.write(" ")
		file.write(stringToInt(row['sim_table'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
		file.write('\n')

		file.write(stringToInt(row['paper_id'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
		file.write(" ")
		file.write(stringToInt(row['sim_paper'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
		file.write('\n')
		for w_id in row['paper_word_ind']:
			file.write(stringToInt(row['paper_id'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
			file.write(" ")
			file.write(stringToInt(w_id, maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
			file.write('\n')
		# table word
		for w_id in row['table_word_ind']:
			file.write(stringToInt(row['table_id'], maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
			file.write(" ")
			file.write(stringToInt(w_id, maxPaper, maxTable, dict_ind_table, dict_ind_paper, dict_ind_word))
			file.write('\n')

def sh(script):
    os.system("bash -c '%s'" % script)

def read_embedding(file_dir, maxPaper, maxTable):
	file = open(file_dir, 'r')
	dict_graph_vec = {}
	for line in file:
		list_emb = line.strip().split(" ")
		key = int(list_emb[0])
		if key < maxPaper :
			# paper
			key = "p" + str(key)
		elif key >= maxPaper and key < maxTable + maxPaper:
			# table
			key  =  "t" + str(key - maxPaper)
		else:
			# word
			key  =  "w" + str(key - maxPaper - maxTable)
		vec = []
		for i in range(1, len(list_emb)):
			vec.append(float(list_emb[i]))
		dict_graph_vec[key] = vec
	# print(dict_graph_vec)
	return dict_graph_vec 


def get_dict(list_text, id_val):
	dict_d = {}
	for i in range(0, len(list_text)):
		dict_d[id_val[i]] = list_text[i]
	return dict_d



def main():
	fb.printTable() 
	# fb.main()
	# fb.baseline_methods("entity resolution")
	###################################
	#####Read in paper and caption#####
	###################################
	root_dir = input("Please enter the location of your extracted json file of papers: ")
	# root_dir =  "/Users/zhengshuangjing/desktop/tableDatabase/mergedJson"
	df_ori, maxPaper, maxTable = read_json(root_dir)
	dict_cap_paper = {} # stores the table caption with corresponding paper name
	cap_list = df_ori["table_cap"].values.tolist()
	paper_list = df_ori["paper_name"].values.tolist()
	for i in range(0, len(cap_list)):
		dict_cap_paper[cap_list[i]] = paper_list[i]
	df = df_ori
	path = input("Please enter the location of your papers: ")
	# path =  "/Users/zhengshuangjing/Desktop/tableDatabase/Entity_Rsurvey/"
	paperName, paperContent = fb.readPaper(path + "/")
	dictPaper = {}
	for i in range(len(paperName)):
		dictPaper[paperName[i][0: -4]] = paperContent[i]
	
	list_append_paper = []
	for item in df['paper_name']:
		list_append_paper.append(dictPaper[item])
	
	df["paper_text"] = list_append_paper

	list_corpus = df["paper_text"].values.tolist() + df["table_cap"].values.tolist()
	word_corpus, corpus_word = getWordSet(list_corpus) # word_corpus ==> key = word, value = ind  corpus_word ==> key = ind (w1), value = word

	list_cap_word = []
	for sen in df["table_cap"]:
		sen = fb.sen_clean(sen)
		sen = sen.split(" ")
		word_set = set()
		for word in sen:
			if word in word_corpus:
				word_set.add(word_corpus[word])
		list_cap_word.append(word_set)

	df["table_word_ind"] = list_cap_word

	list_paper_word = []
	for sen in df["paper_text"]:
		sen = fb.sen_clean(sen)
		sen = sen.split(" ")
		word_set = set()
		for word in sen:
			if word in word_corpus:
				word_set.add(word_corpus[word])
		list_paper_word.append(word_set)
	df["paper_word_ind"] = list_paper_word
	# print(df)

	list_sim_table = getMostSim(df["table_cap"].values.tolist(), df["table_id"].values.tolist())
	df["sim_table"] = list_sim_table
	list_sim_paper = getMostSim(df["paper_text"].values.tolist(), df["paper_id"].values.tolist())
	df["sim_paper"] = list_sim_paper
	# print(df)

	dict_ind_table = {}
	dict_ind_paper = {}
	dict_ind_word = {}
	writeFile(dict_ind_table, dict_ind_paper, dict_ind_word, maxPaper, maxTable, df)
	script = """
	cd deepwalk-master
	pip install deepwalk
	deepwalk --format edgelist --input ../paper.edgelist --output ../paper.embeddings
	"""
	sh(script)

	dict_graph_vec = read_embedding("paper.embeddings", maxPaper, maxTable)
	# print(dict_ind_word)
	dict_d_table = get_dict(df["table_cap"].values.tolist(), df['table_id']) # key => t1, value => "Table ....."
	table_id  =  df['table_id'].values.tolist()
	table_cap_list = df['table_cap'].values.tolist()
	vec_table_cap_graph = []
	for id in table_id:
		vec_table_cap_graph.append(dict_graph_vec[id])
	

	###################
	#####Get Query#####
	###################

	while(True):
		userInput = input("Please enter the research area (enter q to quit): ")
		if userInput == "q":
			break
		fb.baseline_methods(userInput, paperName, paperContent, root_dir)
		print("##################################################################################################################")
		print("Method 4: get average query vector")
		if(userInput == "q"): break
		user = userInput.split(" ")
		query_vec = []
		flag = 0
		for word in user:
			if word in word_corpus:
				key = word_corpus[word]
				query_vec.append(dict_graph_vec[key])
			else:
				print("Error: " + word + " is not in database")
				flag = 1
		if flag == 1: continue
		vec = []
		for i in range(0, len(query_vec[0])):
			sum = 0
			for j in range(0, len(query_vec)):
				sum += query_vec[j][i]
			vec.append(sum / len(query_vec))
		dict_index_sim = {}
		for i in range(len(vec_table_cap_graph)):
			table = np.asarray(vec_table_cap_graph[i])
			sim = fb.similarity(table, np.asarray(vec))
			dict_index_sim[table_id[i]] = sim
		
		sorted_by_value = sorted(dict_index_sim.items(), key=lambda kv: -1 * kv[1])
		sorted_by_value  = sorted_by_value[0:10]
		# print(sorted_by_value)
		for item in sorted_by_value:
			print(dict_d_table[item[0]])
		print(" ")
		for item in sorted_by_value:
			print(dict_cap_paper[dict_d_table[item[0]]])
		print("##################################################################################################################")
		print("Method 5: re-rank tfidf result based on graph embedding")
		
		# # print(sorted_by_value_cap)
		sorted_by_value_cap, dict_index_sim_cap = fb.get_sentence_vector_simlarity_tfidf(df["table_cap"].values, userInput)
		count = 0
		dict_method2 = {}
		for tup in sorted_by_value_cap:
			if(count >= 10): break
			key = "t" + str(tup[0]) #t1
			# print(dict_d_table[key])
			sim = fb.similarity(np.asarray(vec), np.asarray(dict_graph_vec[key]))
			dict_method2[key] = sim
			count += 1
		sorted_by_value_m2 = sorted(dict_method2.items(), key=lambda kv: -1 * kv[1])
		# print(sorted_by_value_m2)
		for tup in sorted_by_value_m2:
			print(dict_d_table[tup[0]])
		print(" ")
		for tup in sorted_by_value_m2:
			print(dict_cap_paper[dict_d_table[tup[0]]])
		# print(new_cap_rank_column)



	# print(df)

	# print(dict_ind_table)
	# print(" ")
	# print(dict_ind_paper)
	# print(" ")
	# print(dict_ind_word)

	# """
	# corpus_word is the dictionary that stores ==> key = w1, value = word
	# dict_ind_word is ==> key = 1, value = w1
	# dict_ind_paper ==> key = 1, value = p1
	# dict_ind_table ==> key = 1, value = t1
	# """








# t-p
# p-p
# t-t


if __name__ == '__main__':
	main()