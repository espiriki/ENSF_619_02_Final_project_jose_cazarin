import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel, RobertaModel
from torchvision.models import *
from transformers import BertTokenizer, RobertaTokenizer
import sys
from random import randint
import random


def MBNetLarge(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = mobilenet_v3_large(weights=_weights)

    # 3 here removes the last linear layer, and gets the dropout output
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(3)])

    # if transfer_learning:
    #     for param in model.parameters():
    #         param.requires_grad = False

    return model


def decision(probability):
    value = random.random()
    print("value: ", value)
    return random.random() < probability


class SpamClassifierDistilBert(nn.Module):
    def __init__(self, model_name, n_classes, drop_ratio):
        super(SpamClassifierDistilBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/distilbert/modeling_distilbert.py#L729
    def forward(self, input_ids, attention_mask):
        distilbert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # dim = HIDDEN_SIZE
        # bs = BATCH_SIZE
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        drop_output = self.drop(pooled_output)

        return self.out(drop_output)


class SpamClassifierBert(nn.Module):
    def __init__(self, model_name, n_classes, drop_ratio):
        super(SpamClassifierBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    # THIS WORKS FOR BERT
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]  # 1 here is getting an array of the shape BATCH_SIZE, HIDDEN_SIZE

        return self.out(bert_output)


class SpamClassifierRoberta(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(SpamClassifierRoberta, self).__init__()
        self.bert = RobertaModel.from_pretrained("roberta-base")
        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = roberta_output[0]
        pooled_output = hidden_state[:, 0]
        drop_output = self.drop(pooled_output)
        return self.out(drop_output)


class RobertaAndMBNet(nn.Module):

    def __init__(self, n_classes, drop_ratio):
        super(RobertaAndMBNet, self).__init__()

        self.text_model = RobertaModel.from_pretrained("roberta-base")
        self.image_model = MBNetLarge(n_classes, True)

        self.drop = nn.Dropout(p=drop_ratio)
        self.fc_layer_neurons = 512

        self.features_to_class_text = \
            nn.Linear(self.text_model.config.hidden_size, n_classes)

        self.features_to_class_image = \
            nn.Linear(self.image_model.classifier[0].out_features, n_classes)

        # 1280 from image + 768 from text
        # 512 is the size of the FC layer
        self.concat_layer = \
            nn.Linear(self.image_model.classifier[0].out_features +
                      self.text_model.config.hidden_size, self.fc_layer_neurons)

        # FC layer to classes
        self.fc_layer = \
            nn.Linear(self.fc_layer_neurons, n_classes)

        self.image_dropout = nn.Dropout2d(p=1.0)
        self.text_dropout = nn.Dropout1d(p=1.0)
        self.prob_image_text_dropout = 0.33

    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text
        if eval:
            self.image_dropout.train()
            self.text_dropout.train()
            if remove_image:
                print("    zeroing images on EVAL")
                _images = self.image_dropout(_images)
            if remove_text:
                print("    zeroing text on EVAL")
                _input_ids = self.text_dropout(_input_ids)
                
            if not remove_image and not remove_text:
                print("using both on EVAL")                

        else:
            if decision(self.prob_image_text_dropout):
                image_or_text = decision(0.5)
                if image_or_text:
                    print("    zeroing images")
                    _images = self.image_dropout(_images)
                else:
                    print("    zeroing text")
                    _input_ids = self.text_dropout(_input_ids)
            else:
                print("using both")

        # cur_image = _images[0, :, :, :]
        # print("image shape: ", cur_image.shape)
        # print("    first 10 elements of first channel {}".format(
        #     cur_image[0, :, :][:10]))
        # print("    first 10 elements of second channel {}".format(
        #     cur_image[1, :, :][:10]))
        # print("    first 10 elements of third channel {}".format(
        #     cur_image[2, :, :][:10]))

        # cur_text = _input_ids[0, :]
        # print("text shape: ", cur_text.shape)
        # print("    first 10 elements of the text channel {}".format(
        #     cur_text[:10]))

        text_output = self.text_model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )
        hidden_state = text_output[0]
        text_features = hidden_state[:, 0]

        image_features = self.image_model(_images)

        # print("Image Features Output Shape: {}".format(
        #     image_features.shape))
        # image_model_out_classes = self.features_to_class_image(
        #     image_output_features)
        # print("Image Class Output Shape: {}".format(
        #     image_model_out_classes.shape))

        # print("Text Features Output Shape: {}".format(text_features.shape))
        # text_model_out_classes = self.features_to_class_text(pooled_output)
        # print("Text Class Output Shape: {}".format(
        #     text_model_out_classes.shape))

        image_plus_text_features = torch.cat(
            (text_features, image_features), dim=1)

        # print("Image+Text Features Shape: {}".format(
        #     image_plus_text_features.shape))

        after_concat = self.concat_layer(image_plus_text_features)
        after_drop = self.drop(after_concat)
        final_output = self.fc_layer(after_drop)
        # print("Image+Text Class Shape: {}".format(
        #     final_output.shape))

        return final_output

    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained('roberta-base')

    def get_image_size(self):
        return (320, 320)
