import os
import sys
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertConfig

import sys

BASE_DIR = 'mnt/jessica/ngillani/school_reviews_2.0/'

sys.path.append("{}src/models/base/".format(BASE_DIR))
from bert_models import MeanBertForSequenceRegression, RobertForSequenceRegression

class AdaptedMeanBertForSequenceRegression(nn.Module):
        def __init__(self, config, hid_dim=768, num_output=1):
                super(AdaptedMeanBertForSequenceRegression, self).__init__()
                self.config = config
                self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
                for name, param in self.bert.named_parameters():
                        if 'layer.11' not in name and 'pooler' not in name:
                                param.requires_grad=False
                        # param.requires_grad = False

                self.fc1 = nn.Linear(config.hidden_size, hid_dim)
                self.relu = torch.nn.ReLU()
                self.output_layer = nn.Linear(hid_dim, num_output)

                self.dropout = nn.Dropout(config.hidden_dropout_prob)

        '''
                input_ids = n_sent x max_len
        '''
        def forward(self, input_ids, attention_mask=None):
                outputs = self.bert(input_ids, attention_mask=attention_mask) # [n_sent, dim]
                sent_embs = self.dropout(outputs[0].mean(dim=1)) # [n_sent, config.hidden_size]
                sent_embs = sent_embs.mean(dim=0) # [1, config.hidden_size]
                return self.output_layer(self.relu(self.fc1(sent_embs)))

def bert_forward_wrapper(input_ids, attention_mask=None, position=0):
    return model(input_ids, attention_mask=attention_mask)

def normalize_attributions(attributions, percentile):
    curr_attributions = attributions.cpu().numpy()
    vmax = np.percentile(curr_attributions, percentile)
    vmin = np.min(curr_attributions)
#     normalized_attributions = np.clip((curr_attributions - vmin) / (vmax - vmin), 0, 1)
    normalized_attributions = (curr_attributions - vmin) / (vmax - vmin)
    return torch.Tensor(normalized_attributions)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

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


def compute_and_output_attributions():
	
	Print 
	import pickle
	prepared_data_file = '{}data/Parent_gs_comments_by_school_mn_avg_eb_1.7682657723517046.p'.format(BASE_DIR)

	with open(prepared_data_file, 'rb') as f:
	     all_input_ids, labels_test_score, attention_masks, sentences_per_school = pickle.load(f, encoding='latin1')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# device = "cpu"
	print(device)
	print (torch.cuda.is_available())

	# Load model
	# dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb
	MODEL_DIR = 'runs/bert_reviews/Mar15_2020/mn_avg_eb/'
	BEST_MODEL_DIR = 'dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb/'
	model_path = '{}{}{}e7_loss1.0341.pt'.format(BASE_DIR, MODEL_DIR, BEST_MODEL_DIR)
	dropout_prob = BEST_MODEL_DIR.split('dropout_')[1].split('-')[0]
	config = BertConfig(output_attentions=True, hidden_dropout_prob=dropout_prob, attention_probs_dropout_prob=dropout_prob)
	hidden_dim = BEST_MODEL_DIR.split('hid_dim_')[1].split('-')[0]
	model = AdaptedMeanBertForSequenceRegression(config, hid_dim=hidden_dim, num_output=1)

	# Load state dict and do some post-processing to map variable names correctly
	state_dict = torch.load(model_path, map_location=torch.device('cpu'))
	from collections import OrderedDict
	updated_state_dict = OrderedDict()
	for k in state_dict:
	    curr_key = k
	    if curr_key.startswith(('model.bert', 'model.fc1', 'model.output_layer')):
	        curr_key = curr_key.split('model.')[1]
	    updated_state_dict[curr_key] = state_dict[k]
	    
	model.load_state_dict(updated_state_dict)

	model.to(device)
	model.eval()
	model.zero_grad()

	# load tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# Should be loading from model_path, but again, model wasn't saved with huggingface's save...() function
	# tokenizer = BertTokenizer.from_pretrained(model_path)

	from captum.attr import TokenReferenceBase
	from captum.attr import IntegratedGradients, LayerIntegratedGradients
	from captum.attr import visualization as viz

	data_splits = ['validation', 'train']
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

	#         if count == 1: break
	        count += 1

	        # Prepare data
	        input_ids = torch.LongTensor([all_input_ids[d][i]]).squeeze(0).to(device)
	        label_t = torch.tensor([labels_test_score[d][i]]).to(device)
	        input_mask = torch.tensor([attention_masks[d][i]]).squeeze(0).to(device)
	        
	        pred = model(input_ids, attention_mask=input_mask)
	        mse = (pred.item() - label_t.item()) ** 2
	        
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

	        # Summarize attributions and output
	        summarized_attr = summarize_attributions(attributions).squeeze(0)
	        n_sent = summarized_attr.size(0)
	        attr_for_school_sents = defaultdict(dict)
	        for j in range(0, n_sent):
	            indices = input_ids[j].detach().squeeze(0).tolist()
	            all_tokens = tokenizer.convert_ids_to_tokens(indices)
	            attr_for_school_sents[j]['tokens'] = all_tokens
	            attr_for_school_sents[j]['attributions'] = summarized_attr[j].tolist()
	            assert (len(attr_for_school_sents[j]['tokens']) == len(attr_for_school_sents[j]['attributions']))
	#         print (json.dumps(attr_for_school_sents, indent=4))
	        f = open(OUTPUT_FILE.format(BASE_DIR, BEST_MODEL_DIR, i, d, mse), 'w')
	        f.write(json.dumps(attr_for_school_sents, indent=4))
	        f.close()
	        
	#        all_summarized_attr.append(summarize_attributions(attributions).squeeze(0))
	#        input_ids_for_attr.append(input_ids)


if __name__ == "__main__":
	compute_and_output_attributions()