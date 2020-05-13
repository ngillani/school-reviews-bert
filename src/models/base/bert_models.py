import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel
import pdb
import numpy as np

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


class RobertForSequenceRegression(nn.Module):
	def __init__(self, config, num_output=1, recurrent_hidden_size=1024, recurrent_num_layers=1):
		super(RobertForSequenceRegression, self).__init__()
		self.config = config
		self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
		for name, param in self.bert.named_parameters():
			if 'layer.11' not in name and 'pooler' not in name:
				param.requires_grad=False
			# param.requires_grad = False

		self.num_output = num_output

		self.gru = torch.nn.GRU(config.hidden_size, recurrent_hidden_size, recurrent_num_layers, batch_first=True)

		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.fc1 = nn.Linear(recurrent_hidden_size, recurrent_hidden_size)
		self.output_layer = nn.Linear(recurrent_hidden_size, 1)

		if num_output > 1:
			self.fc_confounds = nn.Linear(recurrent_hidden_size, recurrent_hidden_size)
			self.output_layer_confounds = nn.Linear(recurrent_hidden_size, num_output - 1)

		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		print ("Number of model params", sum([np.prod(p.size()) for p in model_parameters]))


	'''
		input_ids = n_schools x n_sent x max_len
		sents_per_school = tensor of ints per school
	'''
	def forward(self, input_ids, sents_per_school, attention_mask=None):
		n_schools, n_sent, max_len = input_ids.size()
		inputs = input_ids.view(-1, max_len) # [n_schools * n_sent, max_len]
		
		if attention_mask is not None:
			attends = attention_mask.view(-1, max_len)
			outputs = self.bert(inputs, attention_mask=attends) # [n_schools * n_sent, dim]
		else:
			outputs = self.bert(inputs)

		sent_embs = self.dropout(outputs[0].mean(dim=1)) # [n_schools * n_sent, config.hidden_size]
		sent_embs = sent_embs.view(n_schools, n_sent, sent_embs.size(-1))
		packed_sent_embs = torch.nn.utils.rnn.pack_padded_sequence(sent_embs, sents_per_school,
																   batch_first=True)
		recurrent_output = self.gru(packed_sent_embs)[1].squeeze(0) # [n_schools, recurrent_hidden_size]
		
		confounds_pred = None
		target_pred = self.output_layer(self.relu(self.fc1(recurrent_output)))
	
		if self.num_output > 1:
			recurrent_output = grad_reverse(recurrent_output)
			confounds_pred = self.output_layer_confounds(self.relu(self.fc_confounds(recurrent_output)))
	
		return target_pred, confounds_pred


class MeanBertForSequenceRegression(nn.Module):
	def __init__(self, config, hid_dim=768, num_output=1):
		super(MeanBertForSequenceRegression, self).__init__()
		self.config = config
		self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
		for name, param in self.bert.named_parameters():
			if 'layer.11' not in name and 'pooler' not in name:
				param.requires_grad=False
			# param.requires_grad = False

		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		self.num_output = num_output
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(config.hidden_size, hid_dim)
		self.output_layer = nn.Linear(hid_dim, 1)

		if num_output > 1:
			self.fc_confounds = nn.Linear(config.hidden_size, hid_dim)
			self.output_layer_confounds = nn.Linear(hid_dim, num_output - 1)

		# model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		# print ("Number of model params", sum([np.prod(p.size()) for p in model_parameters]))


	'''
		input_ids = n_schools x n_sent x max_len
	'''
	def forward(self, input_ids, attention_mask=None):
		n_schools, n_sent, max_len = input_ids.size()
		inputs = input_ids.view(-1, max_len) # [n_schools * n_sent, max_len]

		if attention_mask is not None:
			attends = attention_mask.view(-1, max_len)
			outputs = self.bert(inputs, attention_mask=attends) # [n_schools * n_sent, dim]
		else:
			outputs = self.bert(inputs)

		sent_embs = self.dropout(outputs[0].mean(dim=1)) # [n_schools * n_sent, config.hidden_size]
		sent_embs = sent_embs.view(n_schools, n_sent, sent_embs.size(-1))
		sent_embs = sent_embs.mean(dim=1) # [n_schools, config.hidden_size]
		
		confounds_pred = None
		target_pred = self.output_layer(self.relu(self.fc1(sent_embs)))
	
		if self.num_output > 1:
			sent_embs = grad_reverse(sent_embs)
			confounds_pred = self.output_layer_confounds(self.relu(self.fc_confounds(sent_embs)))
	
		return target_pred, confounds_pred


class BertEncoder(nn.Module):
	def __init__(self, config, num_output=1):
		super(BertEncoder, self).__init__()
		self.config = config
		self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)

	'''
		input_ids = n_schools x n_sent x max_len
	'''
	def forward(self, input_ids, attention_mask=None):
		n_schools, n_sent, max_len = input_ids.size()
		inputs = input_ids.view(-1, max_len) # [n_schools * n_sent, max_len]

		if attention_mask is not None:
			attends = attention_mask.view(-1, max_len)
			outputs = self.bert(inputs, attention_mask=attends) # [n_schools * n_sent, dim]
		else:
			outputs = self.bert(inputs)

		sent_embs = outputs[0].mean(dim=1) # [n_schools * n_sent, config.hidden_size]
		sent_embs = sent_embs.view(n_schools, n_sent, sent_embs.size(-1))
		sent_embs = sent_embs.mean(dim=1) # [n_schools, config.hidden_size]

		return sent_embs


class BertEncoderForComments(nn.Module):
	def __init__(self, config, num_output=1):
		super(BertEncoderForComments, self).__init__()
		self.config = config
		self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)

	'''
		input_ids = n_comments x max_len
	'''
	def forward(self, input_ids, attention_mask=None):
		outputs = self.bert(input_ids, attention_mask=attention_mask) # [n_comments, max_len, config.hidden_size]
		sent_embs = outputs[0].mean(dim=1) # [n_comments, config.hidden_size]

		return sent_embs # [n_comments, config.hidden_size]


if __name__ == "__main__":
		from transformers import BertConfig
		config = BertConfig(output_attentions=True)
		robert = RobertForSequenceRegression(config)
		loss_fct = StraightUpLoss()        
		robert.cuda()
		n_schools = 4
		n_sent = 50
		max_len = 64
		input_ids = torch.zeros(n_schools, n_sent, max_len).long().cuda()
		sents_per_school = torch.tensor([50, 48, 46, 3]).cuda()
		output = robert(input_ids, sents_per_school)
		t_loss = loss_fct.compute_loss(output, torch.tensor([10.0, 2.0, 3.0, 1.0]).cuda())
		t_loss.backward()
