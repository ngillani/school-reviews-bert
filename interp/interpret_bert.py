print ('importing packages ...')
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertConfig

import sys
sys.path.append("/home/ubuntu/school_reviews/school_reviews_bert/src/models/base/")
from bert_models import MeanBertForSequenceRegression, RobertForSequenceRegression

print ('defining helper functions ...')
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

print ('loading data ...')
import pickle
BATCH_SIZE = 1
prepared_data_file = '/home/ubuntu/school_reviews/school_reviews_bert/data/Parent_gs_comments_by_school_mn_avg_eb_1.7682657723517046.p'

with open(prepared_data_file, 'rb') as f:
     all_input_ids, labels_test_score, attention_masks, sentences_per_school = pickle.load(f, encoding='latin1')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)
print (torch.cuda.is_available())

print ('Loading model ...')
# dropout_0.3-hid_dim_256-lr_0.0001-model_type_meanbert-outcome_mn_avg_eb
model_path = '/home/ubuntu/school_reviews/school_reviews_bert/saved_models/e7_loss1.0341.pt'
model_MSE = float(model_path.split('loss')[1].split('.pt')[0])

config = BertConfig(output_attentions=True, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
model = MeanBertForSequenceRegression(config, hid_dim=256, num_output=1)
sys.path.append("/home/ubuntu/school_reviews/school_reviews_bert/src/models/base/")
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

print ('Running integrated gradients ...')
# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Should be loading from model_path, but again, model wasn't saved with huggingface's save...() function
# tokenizer = BertTokenizer.from_pretrained(model_path)

from captum.attr import TokenReferenceBase
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz

data_splits = ['validation']
all_viz = []
count = 0

batch_size = 1
internal_batch_size = 16
n_steps = 2

for d in data_splits:

    n_schools = torch.LongTensor(all_input_ids[d]).size(0)
    
    for i in range(0, n_schools):

        if i == 1: break

        count += 1

        # Prepare data
        input_ids = torch.LongTensor([all_input_ids[d][i]]).to(device)
        label_t = torch.tensor([labels_test_score[d][i]]).to(device)
        input_mask = torch.tensor([attention_masks[d][i]]).to(device)

        print (input_ids.size())
        print (label_t.size())
        print (input_mask.size())

        # Generate reference sequence for integrated gradients
        ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
        token_reference = TokenReferenceBase(reference_token_idx=ref_token_id)
        ref_input_ids = token_reference.generate_reference(input_ids.size(2), device=device).unsqueeze(0).unsqueeze(1).repeat(1, input_ids.size(1), 1).long()
        print ('ref size: ', ref_input_ids.size())

        # Compute integrated gradients
        lig = LayerIntegratedGradients(bert_forward_wrapper, model.bert.embeddings)
        attributions, conv_delta = lig.attribute(
            inputs=input_ids, 
            baselines=ref_input_ids,
            additional_forward_args=(input_mask, 0),  
#                 internal_batch_size=internal_batch_size,
            n_steps=n_steps,
            return_convergence_delta=True)

        print ('After IG is run!!!')
        attributions_sum = summarize_attributions(attributions).squeeze(0)
        indices = input_ids[0].detach().squeeze(0).tolist()
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

        all_viz.append(vis)


