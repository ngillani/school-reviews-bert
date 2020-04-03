"""
Usage:
PYTHONPATH=. python src/models/bert_reviews.py --hid_dim 128
"""

from datetime import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


from transformers import BertModel, BertConfig#, BertAdam
# from pytorch_pretrained_bert import BertAdam

from config import RUNS_PATH
from src.models.base.bert_models import RobertForSequenceRegression, MeanBertForSequenceRegression
from src.models.core import experiments, nn_utils
from src.models.core.train_nn import TrainNN
from src.models.dataset import make_dataloader, load_and_cache_data


USE_CUDA = torch.cuda.is_available()



##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        # Training
        self.batch_size = 16
        self.lr = 0.0001  # 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 12

        # Data
        self.max_len = 30

        # Model
        self.dropout = 0.1  # 0.1, 0.2
        self.hid_dim = 768  # 256, 768
        self.n_layers = 1
        self.outcome = 'mn_avg_eb'  # mn_avg_eb, mn_grd_eb, top_level
        # self.adv_outcome = ''
        self.model_type = 'meanbert'  # 'robert' 'meanbert


class BertReviewsModel(TrainNN):
    def __init__(self, hp, save_dir=None):
        super(BertReviewsModel, self).__init__(hp, save_dir)

        input_ids, labels_test_score, attention_masks, num_sentences_per_school = load_and_cache_data(
            outcome=hp.outcome, max_len=hp.max_len)
        self.tr_loader = make_dataloader(
            (input_ids['train'], attention_masks['train'], labels_test_score['train'], num_sentences_per_school['train']),
            hp.batch_size)

        self.val_loader  = make_dataloader(
            (input_ids['validation'],
             attention_masks['validation'], labels_test_score['validation'], num_sentences_per_school['validation']),
            hp.batch_size)

        # Model 
        config = BertConfig(output_attentions=True, hidden_dropout_prob=hp.dropout, attention_probs_dropout_prob=hp.dropout)
        # TODO: set num_outputs depending on number of outcomes / adv_outcome?
        if hp.model_type == 'meanbert':
            self.model = MeanBertForSequenceRegression(config, hid_dim=hp.hid_dim, num_output=1)
        elif hp.model_type == 'robert':
            self.model = RobertForSequenceRegression(config, num_output=1,
                recurrent_hidden_size=hp.hid_dim, recurrent_num_layers=hp.n_layers)
        
        self.models.append(self.model)
        for model in self.models:
            model.cuda()

        # Optimizers
        # TODO: use BertADAM?
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def compute_loss(self, predicted_t, actual_t):
        t_loss = F.mse_loss(predicted_t, actual_t)
        return t_loss

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass
    
        Returns: dict: 'loss': float Tensor must exist
        """
        
        input_ids, input_mask, test_scores, num_sentences_per_school = batch
        num_sentences_per_school, perm = torch.sort(num_sentences_per_school, descending=True)
        input_ids =  nn_utils.move_to_cuda(input_ids[perm, :, :])
        input_mask =  nn_utils.move_to_cuda(input_mask[perm, :, :])
        test_scores =  nn_utils.move_to_cuda(test_scores[perm].unsqueeze_(1))
        num_sentences_per_school =  nn_utils.move_to_cuda(num_sentences_per_school)
                
        if self.hp.model_type == 'meanbert':
    	    predicted = self.model(input_ids, attention_mask=input_mask)  # [bsz] (n_outcomes)
        elif self.hp.model_type == 'robert':
            predicted = self.model(input_ids, num_sentences_per_school, attention_mask=input_mask)  # [bsz] (n_outcomes)

        loss = self.compute_loss(predicted, test_scores)
    
        return {'loss': loss}


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'bert_reviews', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
    experiments.save_run_data(save_dir, hp)

    model = BertReviewsModel(hp, save_dir)

    model.train_loop()
