import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertConfig
# from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients, IntegratedGradients, LayerConductance, TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

import sys
sys.path.append("/media/roy/ngillani/school_ratings/models")
from robert_regressor import MeanBertForSequenceRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
# model_path = '<PATH-TO-SAVED-MODEL>'
model_path = '/media/roy/ngillani/school_ratings/models/checkpoints/top_level/max_len_30_lr_0.0001/epoch_1_training_loss_0.328397995247_val_loss_0.332628061866.pt'

# HERE: https://github.com/huggingface/transformers/issues/2094#issuecomment-563346322
# 1) Using hugginface's save_model, can then call from_pretrained where model_path is the directory
# 2) did some version of torch.save()
	# config = BertConfig.from_pretrained("bert-base-cased", num_labels=3)
	# model = BertForSequenceClassification.from_pretrained("bert-base-cased", config=config)
	# model.load_state_dict(torch.load("SAVED_SST_MODEL_DIR/pytorch_model.bin"))

# Version 2
config = BertConfig(output_attentions=True)
# model = BertModel.from_pretrained('bert-base-uncased', config=config)
model = MeanBertForSequenceRegression(config)
sys.path.append('/media/roy/ngillani/school_ratings/models')  # model being saved requires bert_regresser.py to be in path
state_dict = torch.load(model_path, map_location=torch.device('cpu')).state_dict()  # currently loads BertForSequenceRegression
model.load_state_dict(state_dict)


# model = BertForQuestionAnswering.from_pretrained(model_path)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Should be loading from model_path, but again, model wasn't saved with huggingface's save...() function
# tokenizer = BertTokenizer.from_pretrained(model_path)

# Load data
import pickle
BATCH_SIZE = 1
prepared_data_file = '/media/roy/ngillani/school_ratings/data/tiny_by_school_top_level.p'

with open(prepared_data_file, 'rb') as f:
	 all_input_ids, labels_test_score, attention_masks, sentences_per_school = pickle.load(f, encoding='latin1')

TARGET_IND = 0

input_ids = torch.LongTensor([all_input_ids['train'][TARGET_IND]])
label_t = torch.tensor([labels_test_score['train'][TARGET_IND]])
input_mask = torch.tensor([attention_masks['train'][TARGET_IND]])

print (input_ids.size())

preds1 = model(input_ids, attention_mask=input_mask)

ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
token_reference = TokenReferenceBase(reference_token_idx=ref_token_id)
ref_input_ids = token_reference.generate_reference(input_ids.size(2), device=device).unsqueeze(0).unsqueeze(1).repeat(1, 100, 1).long()
print (ref_input_ids.size())

def bert_forward_wrapper(input_ids, attention_mask=None, position=0):
	return model(input_ids, attention_mask=attention_mask)

truth = torch.tensor([label_t.item()]).unsqueeze(0).long()
input_mask = input_mask.long()
input_ids = input_ids.long()
ref_input_ids = ref_input_ids.long()

from captum.attr import LayerIntegratedGradients
from captum.attr import LayerConductance
lig = LayerIntegratedGradients(bert_forward_wrapper, model.bert.embeddings)

attributions, conv_delta = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
	additional_forward_args=(input_mask, 0),  return_convergence_delta=True)

def summarize_attributions(attributions):
	print(attributions)
	print(attributions.size())
	attributions = attributions.sum(dim=-1).squeeze(0)
	attributions = attributions / torch.norm(attributions)
	return attributions

attributions_sum = summarize_attributions(attributions)
indices = input_ids[0].detach().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(indices)

vis = viz.VisualizationDataRecord(
						attributions_sum,
						None,
						None,
						None,
						None,
						attributions_sum.sum(),       
						all_tokens,
						conv_delta)


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

visualize_text([vis])
preds1

from IPython.display import Image
Image(filename='visuals.png')