import os
import sys
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertConfig

import sys

# from utils.header import *

# BASE_DIR = '/home/ubuntu/school_reviews/school_reviews_bert/'
BASE_DIR = '/media/jessica/ngillani/school_ratings_2.0/'
# PREPARED_DATA_FILE_mn_avg_eb = '{}data/Parent_gs_comments_by_school_mn_avg_eb_1.7682657723517046.p'.format(BASE_DIR)
PREPARED_DATA_FILE_mn_avg_eb = '{}data/Parent_gs_comments_by_school_with_covars_mn_avg_eb_1.753468860986852.p'.format(BASE_DIR)
# PREPARED_DATA_FILE_mn_grd_eb = '{}data/Parent_gs_comments_by_school_mn_grd_eb_0.034058608806675876.p'.format(BASE_DIR)
PREPARED_DATA_FILE_mn_grd_eb = '{}data/Parent_gs_comments_by_school_with_covars_mn_grd_eb_0.03407799589996242.p'.format(BASE_DIR)
PREPARED_DATA_FILE_top_level = '{}data/Parent_gs_comments_by_school_with_covars_top_level_1.304894531887436.p'.format(BASE_DIR)
PREPARED_DATA_FILE_mn_avg_eb_adv = '{}data/Parent_gs_comments_by_school_with_covars_mn_avg_eb_1.753468860986852.p'.format(BASE_DIR)
PREPARED_DATA_FILE_top_level_adv = '{}data/Parent_gs_comments_by_school_with_covars_top_level_1.304894531887436.p'.format(BASE_DIR)
# PREPARED_DATA_FILE_mn_avg_eb_adv = '{}data/tiny_Parent_gs_comments_by_school_with_covars_mn_avg_eb_1.8568614088656685.p'.format(BASE_DIR)
#PREPARED_DATA_FILE_top_level = '{}data/tiny_by_school_top_level.p'.format(BASE_DIR)

sys.path.append("{}src/models/base/".format(BASE_DIR))
print(sys.path)
# from bert_models import MeanBertForSequenceRegression, RobertForSequenceRegression


class GradientReverse(torch.autograd.Function):
	scale = 1.0
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
	GradientReverse.scale = scale
	return GradientReverse.apply(x)

class AdaptedMeanBertForSequenceRegression(nn.Module):
		def __init__(self, config, hid_dim=768, num_output=1):
				super(AdaptedMeanBertForSequenceRegression, self).__init__()
				self.config = config
				self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
				for name, param in self.bert.named_parameters():
								if 'layer.11' not in name and 'pooler' not in name:
												param.requires_grad=False
								# param.requires_grad = False

				self.num_output = num_output
				self.fc1 = nn.Linear(config.hidden_size, hid_dim)
				self.relu = torch.nn.ReLU()
				self.output_layer = nn.Linear(hid_dim, 1)

				if num_output > 1:
						self.fc_confounds = nn.Linear(config.hidden_size, hid_dim)
						self.output_layer_confounds = nn.Linear(hid_dim, num_output - 1)

				self.dropout = nn.Dropout(config.hidden_dropout_prob)


		'''
				input_ids = n_sent x max_len
		'''
		def forward(self, input_ids, attention_mask=None):
				outputs = self.bert(input_ids, attention_mask=attention_mask) # [n_sent, dim]
				sent_embs = self.dropout(outputs[0].mean(dim=1)) # [n_sent, config.hidden_size]
				sent_embs = sent_embs.mean(dim=0) # [1, config.hidden_size]

				confounds_pred = None
				target_pred = self.output_layer(self.relu(self.fc1(sent_embs)))
				# target_pred = self.output_layer_confounds(self.relu(self.fc_confounds(sent_embs)))
				return target_pred


class AdaptedRobertForSequenceRegression(nn.Module):
	def __init__(self, config, num_output=1, recurrent_hidden_size=1024, recurrent_num_layers=1):
		super(AdaptedRobertForSequenceRegression, self).__init__()
		self.config = config
		self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
		for name, param in self.bert.named_parameters():
				if 'layer.11' not in name and 'pooler' not in name:
						param.requires_grad=False
				# param.requires_grad = False

		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.fc1 = nn.Linear(recurrent_hidden_size, recurrent_hidden_size)
		self.output_layer = nn.Linear(recurrent_hidden_size, 1)
		self.gru = torch.nn.GRU(config.hidden_size, recurrent_hidden_size, recurrent_num_layers, batch_first=True)

		if num_output > 1:
				self.fc_confounds = nn.Linear(recurrent_hidden_size, recurrent_hidden_size)
				self.output_layer_confounds = nn.Linear(recurrent_hidden_size, num_output - 1)

		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		print ("Number of model params", sum([np.prod(p.size()) for p in model_parameters]))


	'''
		input_ids = n_sent x max_len
	'''
	def forward(self, input_ids, attention_mask=None):

		outputs = self.bert(input_ids, attention_mask=attention_mask) # [n_sent, dim]
		sent_embs = self.dropout(outputs[0].mean(dim=1)) # [n_sent, config.hidden_size]
		sent_embs = sent_embs.unsqueeze(0)
		recurrent_output = self.gru(sent_embs)[1].squeeze() # [recurrent_hidden_size]

		target_pred = self.output_layer(F.relu(self.fc1(recurrent_output))) # [1, num_output]
		return target_pred


def visualize_text(datarecords):
		dom = ["<table width: 100%>"]
		rows = [
				"<th>Attribution Score</th>"
				"<th>Word Importance</th>"
		]
		for datarecord in datarecords:
				rows.append(
						"".join(
								[
										"<tr>",
										viz.format_classname("{0:.2f}".format(datarecord.attr_score)),
										viz.format_word_importances(
												datarecord.raw_input, datarecord.word_attributions
										),
										"<tr>",
								]
						)
				)

		dom.append("".join(rows))
		dom.append("</table>")
		display(viz.HTML("".join(dom)))


def get_best_model(outcome):

		if outcome == 'mn_avg_eb':
				MODEL_DIR = 'runs/bert_reviews/Jun17_2020/mn_avg_eb/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb/'		
				model_path = '{}{}{}e7_loss1.0226.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		elif outcome == 'mn_avg_eb_adv':				
				# MODEL_DIR = 'runs/bert_reviews/May13_2020/adversarial_training/'
				# BEST_MODEL_DIR = 'adv_terms_perfrl_perwht-dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb/'
				# model_path = '{}{}{}e4_loss1.1809.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

				# Income pred
				MODEL_DIR = 'runs/bert_reviews/Jun18_2020/income_pred/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_perfrl/'		
				model_path = '{}{}{}e4_loss0.0675.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

				# # ## Race pred
				# MODEL_DIR = 'runs/bert_reviews/Jun18_2020/race_pred/'
				# BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_perwht/'		
				# model_path = '{}{}{}e4_loss0.0389.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		elif outcome == 'mn_grd_eb':
				# MODEL_DIR = 'runs/bert_reviews/Mar23_2020/mn_grd_eb/'
				# BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_grd_eb/'
				# model_path = '{}{}{}e4_loss0.0326.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

				MODEL_DIR = 'runs/bert_reviews/Jun17_2020/mn_grd_eb/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_grd_eb/'
				model_path = '{}{}{}e3_loss0.0329.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		elif outcome == 'top_level_adv':
				MODEL_DIR = 'runs/bert_reviews/May13_2020/adversarial_training/'
				BEST_MODEL_DIR = 'adv_terms_perfrl_perwht-dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_top_level/'
				model_path = '{}{}{}e2_loss0.4334.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)			
		else:
				MODEL_DIR = 'runs/bert_reviews/May26_2020/limited_five_star/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_top_level/'
				model_path = '{}{}{}e8_loss0.2495.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		dropout_prob = float(BEST_MODEL_DIR.split('dropout_')[1].split('-')[0])
		config = BertConfig(output_attentions=True, hidden_dropout_prob=dropout_prob, attention_probs_dropout_prob=dropout_prob)
		hidden_dim = int(BEST_MODEL_DIR.split('hid_dim_')[1].split('-')[0])

		state_dict = torch.load(model_path, map_location=torch.device('cpu'))
		updated_state_dict = OrderedDict()

		num_output = 1
		
		# Load state dict and do some post-processing to map variable names correctly
		if outcome in ['mn_avg_eb', 'mn_grd_eb', 'mn_avg_eb_adv']:
			print ('Loading adapted mean bert!')
			if outcome == 'mn_avg_eb_adv':
				num_output = 2
			model = AdaptedMeanBertForSequenceRegression(config, hid_dim=hidden_dim, num_output=num_output)
			for k in state_dict:
					curr_key = k
					if curr_key.startswith(('model.bert', 'model.fc1', 'model.output_layer', 'model.fc_confounds', 'model.output_layer_confounds')):
						curr_key = curr_key.split('model.')[1]
					updated_state_dict[curr_key] = state_dict[k]
					
			model.load_state_dict(updated_state_dict)
			return model, BEST_MODEL_DIR

		else:
			num_layers = int(BEST_MODEL_DIR.split('n_layers_')[1].split('-')[0])
			if outcome == 'top_level_adv':
				num_output = 3
			model = AdaptedRobertForSequenceRegression(config, recurrent_hidden_size=hidden_dim, num_output=num_output, recurrent_num_layers=num_layers)
			for k in state_dict:
					curr_key = k
					if curr_key.startswith(('model.bert', 'model.fc1', 'model.gru', 'model.output_layer', 'model.fc_confounds', 'model.gru_confounds', 'model.output_layer_confounds')):
							curr_key = curr_key.split('model.')[1]
					updated_state_dict[curr_key] = state_dict[k]
					
			model.load_state_dict(updated_state_dict)
			return model, BEST_MODEL_DIR


def output_data_to_covar_mapping(
				prepared_data_files=[PREPARED_DATA_FILE_mn_avg_eb_adv],
				output_file='{}data/{}_validation_covar_mapping.json'
		):

		import pickle

		for prepared_data_file in prepared_data_files:
			prepared_data_file = prepared_data_file.format(BASE_DIR)
			with open(prepared_data_file, 'rb') as f:
					all_input_ids, labels_target, attention_masks, sentences_per_school, url, perfrl, perwht, share_singleparent, totenrl, share_collegeplus, mail_returnrate = pickle.load(f, encoding='latin1')

			mapping = defaultdict(dict)
			for i in range(0, len(all_input_ids['validation'])):
				print (prepared_data_file, i)
				mapping[i] = {
					'url': url['validation'][i],
					'perfrl': perfrl['validation'][i],
					'perwht': perwht['validation'][i],
					'share_singleparent': share_singleparent['validation'][i],
					'totenrl': totenrl['validation'][i],
					'share_collegeplus': share_collegeplus['validation'][i],
					'mail_returnrate': mail_returnrate['validation'][i],
				}
			
			f = open(output_file.format(BASE_DIR, prepared_data_file.split('/')[-1]), 'w')
			f.write(json.dumps(mapping, indent=4))
			f.close()



def compute_and_output_attributions(
				outcome='mn_grd_eb'
		):

		import pickle

		print ('Loading data ...')
		
		if outcome == 'top_level':
				prepared_data_file = PREPARED_DATA_FILE_top_level
		elif outcome == 'mn_avg_eb':
				prepared_data_file = PREPARED_DATA_FILE_mn_avg_eb
		elif outcome == 'mn_avg_eb_adv':
				prepared_data_file = PREPARED_DATA_FILE_mn_avg_eb_adv
		elif outcome == 'top_level_adv':
				prepared_data_file = PREPARED_DATA_FILE_top_level_adv
		else:
				prepared_data_file = PREPARED_DATA_FILE_mn_grd_eb

		with open(prepared_data_file, 'rb') as f:
				 all_input_ids, labels_target, attention_masks, sentences_per_school, url, perwht, perfrl, share_singleparent, totenrl, share_collegeplus, mail_returnrate = pickle.load(f, encoding='latin1')

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# device = "cpu"
		print(device)
		print (torch.cuda.is_available())

		print ('Loading model ...')
		model, BEST_MODEL_DIR = get_best_model(outcome)

		model.to(device)
		# model.eval()
		model.zero_grad()

		# load tokenizer
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# Should be loading from model_path, but again, model wasn't saved with huggingface's save...() function
		# tokenizer = BertTokenizer.from_pretrained(model_path)

		# Define wrapper function for integrated gradients
		def bert_forward_wrapper(input_ids, attention_mask=None, position=0):
				return model(input_ids, attention_mask=attention_mask)

		from captum.attr import TokenReferenceBase
		from captum.attr import IntegratedGradients, LayerIntegratedGradients
		from captum.attr import visualization as viz

		data_splits = ['validation']
		all_summarized_attr = []
		input_ids_for_attr = []
		count = 0

		internal_batch_size = 12
		n_steps = 48

		OUTPUT_DIR = '{}interp/attributions/{}/'
		OUTPUT_FILE = OUTPUT_DIR + '{}_{}_loss_{}.json'
		if not os.path.exists(OUTPUT_DIR.format(BASE_DIR, BEST_MODEL_DIR)):
			os.makedirs(OUTPUT_DIR.format(BASE_DIR, BEST_MODEL_DIR))

		start_ind = len([int(f.split('_')[0]) for f in os.listdir(OUTPUT_DIR.format(BASE_DIR, BEST_MODEL_DIR))])

		for d in data_splits:

				n_schools = torch.LongTensor(all_input_ids[d]).size(0)
				print ("num schools {} for {} split".format(n_schools, d))
				
				for i in range(start_ind, n_schools):
						
						print (d, i)

		#				  if count == 1: break
						count += 1

						
						# Prepare data
						input_ids = torch.LongTensor([all_input_ids[d][i]]).squeeze(0).to(device)
						label_t = torch.tensor([labels_target[d][i]]).to(device)
						input_mask = torch.tensor([attention_masks[d][i]]).squeeze(0).to(device)
						label_perfrl = torch.tensor([perfrl[d][i]]).to(device)
						label_perwht = torch.tensor([perwht[d][i]]).to(device)
						lable_share_singleparent = torch.tensor([share_singleparent[d][i]]).to(device)
						label_totenrl = torch.tensor([totenrl[d][i]]).to(device)
						label_share_collegeplus = torch.tensor([share_collegeplus[d][i]]).to(device)
						label_mail_returnrate = torch.tensor([mail_returnrate[d][i]]).to(device)

						pred = model(input_ids, attention_mask=input_mask)								
						# print ('pred and actual: ', pred, label_t)
						mse = F.mse_loss(pred[0].unsqueeze_(0), label_t)
						# mse = F.mse_loss(pred[0].unsqueeze_(0), label_perfrl)
						# mse = F.mse_loss(pred[0].unsqueeze_(0), label_perwht)
						if pred.size()[0] > 1:
								mse_perfrl = F.mse_loss(pred[1].unsqueeze_(0), label_perfrl)
								mse_perwht = F.mse_loss(pred[2].unsqueeze_(0), label_perwht)
								mse = '_'.join([str(loss) for loss in [mse, mse_perfrl, mse_perwht]])
						
						# print ('MSE: ', mse)
						# Generate reference sequence for integrated gradients
						ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
						token_reference = TokenReferenceBase(reference_token_idx=ref_token_id)
						ref_input_ids = token_reference.generate_reference(input_ids.size(0), device=device).unsqueeze(1).repeat(1, input_ids.size(1)).long()

						# Compute integrated gradients
						lig = LayerIntegratedGradients(bert_forward_wrapper, model.bert.embeddings)
						attributions, conv_delta = lig.attribute(
								inputs=input_ids, 
								baselines=ref_input_ids,
								additional_forward_args=(input_mask, 0), 
								internal_batch_size=internal_batch_size,
								n_steps=n_steps,
								return_convergence_delta=True)

						# print ('size of attributions: ', attributions.size())
						# Summarize attributions and output
						# summarized_attr = summarize_attributions(attributions).squeeze(0)
						summarized_attr = attributions.sum(dim=-1).squeeze(0)
						# print ('updated attr size: ', summarized_attr.size())
						n_sent = summarized_attr.size(0)
						attr_for_school_sents = defaultdict(dict)
						for j in range(0, n_sent):
								indices = input_ids[j].detach().squeeze(0).tolist()
								all_tokens = tokenizer.convert_ids_to_tokens(indices)
								attr_for_school_sents[j]['tokens'] = all_tokens
								attr_for_school_sents[j]['attributions'] = summarized_attr[j].tolist()
								assert (len(attr_for_school_sents[j]['tokens']) == len(attr_for_school_sents[j]['attributions']))
						f = open(OUTPUT_FILE.format(BASE_DIR, BEST_MODEL_DIR, i, d, mse), 'w')
						f.write(json.dumps(attr_for_school_sents, indent=4))
						f.close()


if __name__ == "__main__":
	compute_and_output_attributions()
	# output_data_to_covar_mapping()
