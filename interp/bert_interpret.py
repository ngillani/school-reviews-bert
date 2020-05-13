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

BASE_DIR = '/mnt/jessica/ngillani/school_ratings_2.0/'
PREPARED_DATA_FILE_mn_avg_eb = '{}data/Parent_gs_comments_by_school_mn_avg_eb_1.7682657723517046.p'.format(BASE_DIR)
PREPARED_DATA_FILE_mn_grd_eb = '{}data/Parent_gs_comments_by_school_mn_grd_eb_0.034058608806675876.p'.format(BASE_DIR)
PREPARED_DATA_FILE_top_level = '{}data/Parent_gs_comments_by_school_top_level_1.3187244708547647.p'.format(BASE_DIR)
PREPARED_DATA_FILE_mn_avg_eb_adv = '{}data/Parent_gs_comments_by_school_with_covars_mn_avg_eb_1.7672375209964608.p'.format(BASE_DIR)
#PREPARED_DATA_FILE_mn_avg_eb_adv = '{}data/tiny_Parent_gs_comments_by_school_with_covars_mn_avg_eb_1.791568987559323.p'.format(BASE_DIR)
#PREPARED_DATA_FILE_top_level = '{}data/tiny_by_school_top_level.p'.format(BASE_DIR)

sys.path.append("{}src/models/base/".format(BASE_DIR))
print(sys.path)
from bert_models import MeanBertForSequenceRegression, RobertForSequenceRegression


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
						self.relu_confounds = nn.ReLU()
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

				if self.num_output > 1:
					sent_embs = grad_reverse(sent_embs)
					confounds_pred = self.output_layer_confounds(self.relu_confounds(self.fc_confounds(sent_embs)))
					# return torch.cat((target_pred, confounds_pred), 0)
					return target_pred
				else:
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
		self.output_layer = nn.Linear(recurrent_hidden_size, num_output)
		self.gru = torch.nn.GRU(config.hidden_size, recurrent_hidden_size, recurrent_num_layers, batch_first=True)

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
		return self.output_layer(F.relu(self.fc1(recurrent_output))) # [1, num_output]


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
				MODEL_DIR = 'runs/bert_reviews/Mar15_2020/mn_avg_eb/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb/'		
				model_path = '{}{}{}e7_loss1.0341.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		elif outcome == 'mn_avg_eb_adv':				
				MODEL_DIR = 'runs/bert_reviews/Apr28_2020/debug/'
				BEST_MODEL_DIR = 'adv_dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb/'
				model_path = '{}{}{}e5_loss1.2159.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		elif outcome == 'mn_grd_eb':
				MODEL_DIR = 'runs/bert_reviews/Mar23_2020/mn_grd_eb/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_grd_eb/'
				model_path = '{}{}{}e4_loss0.0326.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		else:
				# MODEL_DIR = 'runs/bert_reviews/Apr03_2020/top_level/'
				# BEST_MODEL_DIR = 'dropout_0.1-hid_dim_768-lr_0.0001-model_type_robert-n_layers_2-outcome_top_level/'
				# model_path = '{}{}{}e6_loss0.2239.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

				MODEL_DIR = 'runs/bert_reviews/Apr04_2020/top_level/'
				BEST_MODEL_DIR = 'dropout_0.3-hid_dim_768-lr_0.0001-model_type_robert-n_layers_1-outcome_top_level/'
				model_path = '{}{}{}e6_loss0.2270.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)

		dropout_prob = float(BEST_MODEL_DIR.split('dropout_')[1].split('-')[0])
		config = BertConfig(output_attentions=True, hidden_dropout_prob=dropout_prob, attention_probs_dropout_prob=dropout_prob)
		hidden_dim = int(BEST_MODEL_DIR.split('hid_dim_')[1].split('-')[0])

		state_dict = torch.load(model_path, map_location=torch.device('cpu'))
		updated_state_dict = OrderedDict()

		# Load state dict and do some post-processing to map variable names correctly
		if outcome in ['mn_avg_eb', 'mn_grd_eb', 'mn_avg_eb_adv']:
				num_output = 1
				if outcome == 'mn_avg_eb_adv':
						num_output = 3
				model = AdaptedMeanBertForSequenceRegression(config, hid_dim=hidden_dim, num_output=num_output)
				for k in state_dict:
						curr_key = k
						if curr_key.startswith(('model.bert', 'model.fc1', 'model.output_layer', 'model.fc_confounds', 'model.gru_confounds', 'model.output_layer_confounds')):
								curr_key = curr_key.split('model.')[1]
						updated_state_dict[curr_key] = state_dict[k]
						
				model.load_state_dict(updated_state_dict)
				return model, BEST_MODEL_DIR

		else:
				num_layers = int(BEST_MODEL_DIR.split('n_layers_')[1].split('-')[0])
				model = AdaptedRobertForSequenceRegression(config, recurrent_hidden_size=hidden_dim, num_output=1, recurrent_num_layers=num_layers)
				for k in state_dict:
						curr_key = k
						if curr_key.startswith(('model.bert', 'model.fc1', 'model.gru', 'model.output_layer', 'model.fc_confounds', 'model.gru_confounds', 'model.output_layer_confounds')):
								curr_key = curr_key.split('model.')[1]
						updated_state_dict[curr_key] = state_dict[k]
						
				model.load_state_dict(updated_state_dict)
				return model, BEST_MODEL_DIR


def compute_and_output_attributions(
				outcome='mn_avg_eb_adv'
		):

		import pickle

		print ('Loading data ...')
		
		if outcome == 'top_level':
				prepared_data_file = PREPARED_DATA_FILE_top_level
		elif outcome == 'mn_avg_eb':
				prepared_data_file = PREPARED_DATA_FILE_mn_avg_eb
		elif outcome == 'mn_avg_eb_adv':
				prepared_data_file = PREPARED_DATA_FILE_mn_avg_eb_adv
		else:
				prepared_data_file = PREPARED_DATA_FILE_mn_grd_eb

		# TODO(ng): Change this back so we can still load non adversarial data!!!
		with open(prepared_data_file, 'rb') as f:
				 all_input_ids, labels_test_score, perfrl, perwht, attention_masks, sentences_per_school = pickle.load(f, encoding='latin1')

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

		OUTPUT_FILE = '{}interp/attributions/{}/{}_{}_loss_{}.json'

		for d in data_splits:

				n_schools = torch.LongTensor(all_input_ids[d]).size(0)
				print ("num schools {} for {} split".format(n_schools, d))
				
				for i in range(0, n_schools):
						
						print (d, i)

		#				  if count == 1: break
						count += 1

						
						# Prepare data
						input_ids = torch.LongTensor([all_input_ids[d][i]]).squeeze(0).to(device)
						label_t = torch.tensor([labels_test_score[d][i]]).to(device)
						label_perfrl = torch.tensor([perfrl[d][i]]).to(device)
						label_perwht = torch.tensor([perwht[d][i]]).to(device)
						input_mask = torch.tensor([attention_masks[d][i]]).squeeze(0).to(device)

						pred = model(input_ids, attention_mask=input_mask)								
						# print ('pred and actual: ', pred, label_t)
						mse = F.mse_loss(pred[0].unsqueeze_(0), label_t)
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
