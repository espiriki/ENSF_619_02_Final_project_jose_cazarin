import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel, RobertaModel
from torchvision.models import *
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer


class DistilBert(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(DistilBert, self).__init__()
        self.name = "distilbert-base-uncased"
        self.model = DistilBertModel.from_pretrained(self.name)

        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/distilbert/modeling_distilbert.py#L729
    def forward(self, input_ids, attention_mask):
        distilbert_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # dim = HIDDEN_SIZE
        # bs = BATCH_SIZE
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        drop_output = self.drop(pooled_output)

        return self.out(drop_output)

    def get_tokenizer(self):
        return DistilBertTokenizer.from_pretrained(self.name)


class Roberta(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(Roberta, self).__init__()
        self.name = "roberta-base"
        self.model = RobertaModel.from_pretrained(self.name)

        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_state = roberta_output[0]
        pooled_output = hidden_state[:, 0]
        drop_output = self.drop(pooled_output)

        return self.out(drop_output)
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.name)    
