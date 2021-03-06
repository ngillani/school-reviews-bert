import os
import pickle

import spacy
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

import pandas as pd
import pdb
import numpy as np

key_vals = {}	#  key -> set of values

MAX_SENTENCES_PER_SCHOOL = 100

def make_dataloader(data, batch_size, shuffle=True, sampler=None):
	data = TensorDataset(*data)
	data_loader = DataLoader(data, shuffle=shuffle, sampler=sampler, batch_size=batch_size)
	return data_loader

def add_val(key_name, val):
	if key_name not in key_vals:
		key_vals[key_name] = {}
	if val not in key_vals[key_name]:
		key_vals[key_name][val] = len(key_vals[key_name])

def date_to_year(df, split_ind):
#	 print(df['date'][split_ind][0:100])
	return [d.split('-')[0] for d in df['date'][split_ind]]

def load_and_cache_data(
	   raw_data_file='data/Parent_gs_comments_by_school_with_covars.csv',
	   prepared_data_file='data/Parent_gs_comments_by_school_with_covars_%s_%s.p',
	#    raw_data_file='data/tiny_Parent_gs_comments_by_school_with_covars.csv',
	#    prepared_data_file='data/tiny_Parent_gs_comments_by_school_with_covars_%s_%s.p',		 
	   max_len=30,
	   outcome='mn_avg_eb',
	   train_frac = 0.9
	):
	
	print ('Starting load data fcn ...')
	print('Loading data ...')
	
	df = pd.read_csv(raw_data_file).dropna(subset=['url', outcome, 'review_text', 'perfrl', 'perwht', 'singleparent_share2010', 'totenrl', 'frac_coll_plus2010', 'mail_return_rate2010']).reset_index()
	var = np.nanvar(df[outcome])
	prepared_data_file = prepared_data_file % (outcome, var)
	print (prepared_data_file)
	if os.path.isfile(prepared_data_file):
#	 if False:
		with open(prepared_data_file, 'rb') as f:
			input_ids, labels_target, attention_masks, sentences_per_school, url, perfrl, perwht, share_singleparent, totenrl, share_collegeplus, mail_returnrate = pickle.load(f)
		print('data loaded from cache!')
	else:

		all_ind = list(range(0, len(df)))
		np.random.shuffle(all_ind)

		train_ind = all_ind[0:int(train_frac*len(all_ind))]
		val_ind = all_ind[int(train_frac*len(all_ind)):]

		url = {'train': train_ind, 'validation': val_ind}
		data = {'train': list(df['review_text'][train_ind]), 'validation': list(df['review_text'][val_ind])}
		labels_target = {'train': list(df[outcome][train_ind]), 'validation': list(df[outcome][val_ind])}		 
		perwht = {'train': list(df['perwht'][train_ind]), 'validation': list(df['perwht'][val_ind])}
		perfrl = {'train': list(df['perfrl'][train_ind]), 'validation': list(df['perfrl'][val_ind])}
		share_singleparent = {'train': list(df['singleparent_share2010'][train_ind]), 'validation': list(df['singleparent_share2010'][val_ind])}
		totenrl = {'train': list(df['totenrl'][train_ind]), 'validation': list(df['totenrl'][val_ind])}
		share_collegeplus = {'train': list(df['frac_coll_plus2010'][train_ind]), 'validation': list(df['frac_coll_plus2010'][val_ind])}
		mail_returnrate = {'train': list(df['mail_return_rate2010'][train_ind]), 'validation': list(df['mail_return_rate2010'][val_ind])}

		# pd.options.display.max_colwidth = 200
		# print (df['url'][train_ind], df['mn_avg_eb'][train_ind], df['review_text'][train_ind])
		# exit()
		# median_test_score = float(df['mn_avg_eb'][train_ind].median())
		# print("Median test score", median_test_score)
			  
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		spacy_nlp = spacy.load('en_core_web_sm')  # For sentence segmentation

		input_ids = {}	 # split -> list of list of ids
		attention_masks = {}  # split -> list of attention masks
		sentences_per_school = {}  # split -> list of list of number of sentences

		print('Prepping data ...')	  
		for d in data:
			input_ids[d] = []
			attention_masks[d] = []
			sentences_per_school[d] = []
			for review in data[d]:
				try:
					text_sentences = spacy_nlp(review.decode('utf-8'))
					# Choose at most MAX_SENTENCES_PER_SCHOOL to include.
					# Reviews are sorted by recency in the input (newest first).
					token_id_vectors = []
					attention_mask_vectors = []
					for i, sentence in enumerate(text_sentences.sents):
						# Convert to bert IDs
						ids = tokenizer.encode(sentence.text, max_length=max_len)
						# Pad words if needed
						if len(ids) < max_len:
							ids += [0] * (max_len - len(ids))
						attention_mask = [float(id>0) for id in ids]
						token_id_vectors.append(ids)
						attention_mask_vectors.append(attention_mask)
						if i >= MAX_SENTENCES_PER_SCHOOL - 1:
							break
					# Pad sentences if needed
					num_school_sent_before_padding = float(len(token_id_vectors))
					while len(token_id_vectors) < MAX_SENTENCES_PER_SCHOOL:
						token_id_vectors.append([0] * max_len)
						attention_mask_vectors.append([0.0] * max_len)

					input_ids[d].append(token_id_vectors)
					attention_masks[d].append(attention_mask_vectors)
					sentences_per_school[d].append(num_school_sent_before_padding)
				except:
					raise
					pdb.set_trace()

		with open(prepared_data_file, 'wb') as f:
			pickle.dump((input_ids, labels_target, attention_masks, sentences_per_school, url, perfrl, perwht, share_singleparent, totenrl, share_collegeplus, mail_returnrate), f)
			print('Data written to disk')

	# Standardize and tensorize everything
	for d in ['train', 'validation']:
		labels_target[d] = torch.FloatTensor((labels_target[d] - np.mean(df[outcome])) / np.std(df[outcome]))
		perfrl[d] = torch.FloatTensor((perfrl[d] - np.mean(df['perfrl'])) / np.std(df['perfrl']))
		perwht[d] = torch.FloatTensor((perwht[d] - np.mean(df['perwht'])) / np.std(df['perwht']))
		share_singleparent[d] = torch.FloatTensor((share_singleparent[d] - np.mean(df['singleparent_share2010'])) / np.std(df['singleparent_share2010']))
		totenrl[d] = torch.FloatTensor((totenrl[d] - np.mean(df['totenrl'])) / np.std(df['totenrl']))
		share_collegeplus[d] = torch.FloatTensor((share_collegeplus[d] - np.mean(df['frac_coll_plus2010'])) / np.std(df['frac_coll_plus2010']))
		mail_returnrate[d] = torch.FloatTensor((mail_returnrate[d] - np.mean(df['mail_return_rate2010'])) / np.std(df['mail_return_rate2010']))
		
		url[d] = torch.tensor(url[d])
		sentences_per_school[d] = torch.tensor(sentences_per_school[d])
		input_ids[d] = torch.tensor(input_ids[d])
		attention_masks[d] = torch.tensor(attention_masks[d])

	return input_ids, labels_target, attention_masks, sentences_per_school, url, perfrl, perwht, share_singleparent, totenrl, share_collegeplus, mail_returnrate

if __name__ == "__main__":
	load_and_cache_data(outcome='perfrl')
