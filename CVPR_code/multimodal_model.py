import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from torchvision.models import *
from transformers import DistilBertTokenizer
from random import randint
import random

def decision(probability):
    value = random.random()
    return value < probability


def eff_net_v2():

    model = efficientnet_v2_m(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(1)])

    return model

def distilbert():

    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    for param in model.parameters():
        param.requires_grad = False

    return model

class EffV2MediumAndDistilbert(nn.Module):

    def __init__(self, n_classes, drop_ratio, image_text_dropout):
        super(EffV2MediumAndDistilbert, self).__init__()

        self.text_model_name = "distilbert-base-uncased"
        self.text_model = distilbert()
        self.image_model = eff_net_v2()

        self.drop = nn.Dropout(p=drop_ratio)
        self.fc_layer_neurons = 256

        self.image_dropout = nn.Dropout2d(p=1.0)
        self.text_dropout = nn.Dropout1d(p=1.0)
        self.prob_image_text_dropout = image_text_dropout

        # 1280 from image + 768 from text

        # print("OUT OF IMAGE MODEL SHAPE")
        # print(self.image_model.classifier[0].in_features)

        self.image_to_hidden_size = \
            nn.Linear(1280,
                      out_features=self.fc_layer_neurons)

        # print("Hidden size of the distilbert: {}".format(
        #     self.text_model.config.hidden_size))

        self.text_to_hidden_size = \
            nn.Linear(in_features=self.text_model.config.hidden_size,
                      out_features=self.fc_layer_neurons)

        self.concat_layer = \
            nn.Linear(self.fc_layer_neurons*2, self.fc_layer_neurons)

        # FC layer to classes
        self.fc_layer = \
            nn.Linear(self.fc_layer_neurons, n_classes)

        # Layers for gated output
        self.gated_output_hidden_size = 256
        self.hyper_tang_layer = nn.Tanh()
        self.softmax_layer = nn.Softmax(dim=1)

        self.image_features_hidden_layer = \
            nn.Linear(1280,
                      self.gated_output_hidden_size)

        self.text_features_hidden_layer = \
            nn.Linear(self.text_model.config.hidden_size,
                      self.gated_output_hidden_size)

        self.z_layer = \
            nn.Linear(self.gated_output_hidden_size * 2,
                      self.gated_output_hidden_size)

        # FC layer to classes
        self.fc_layer_gated = \
            nn.Linear(self.gated_output_hidden_size, n_classes)

    def forward_normal(self,
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

        text_output = self.text_model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )
        hidden_state = text_output[0]
        text_features = hidden_state[:, 0]

        image_features = self.image_model(_images)

        image_hidden_size = self.image_to_hidden_size(image_features)
        text_hidden_size = self.text_to_hidden_size(text_features)

        image_plus_text_features = torch.cat(
            (image_hidden_size, text_hidden_size), dim=1)

        after_concat = self.concat_layer(image_plus_text_features)
        after_drop = self.drop(after_concat)
        final_output = self.fc_layer(after_drop)

        return final_output

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
                pass                

        else:
            if decision(self.prob_image_text_dropout):
                # choose to drop image or text with equal
                # probabilities
                image_or_text = decision(0.5)
                if image_or_text:
                    print("    zeroing images\n")
                    _images = self.image_dropout(_images)
                else:
                    print("    zeroing text\n")
                    _input_ids = self.text_dropout(_input_ids)
            else:
                print("using both image and text\n")

        text_output = self.text_model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )
        hidden_state = text_output[0]
        text_features = hidden_state[:, 0]

        image_features = self.image_model(_images)

        # 256 * bs
        image_feats_after_tanh =\
            self.hyper_tang_layer(
                self.image_features_hidden_layer(image_features))
        # 256 * bs
        text_feats_after_tanh =\
            self.hyper_tang_layer(
                self.text_features_hidden_layer(text_features))
        # print("image_feats_after_tanh shape: ", image_feats_after_tanh.shape)
        # print("text_feats_after_tanh shape: ", text_feats_after_tanh.shape)

        # 512 * bs
        concat_output_before_tanh = torch.cat(
            (self.image_features_hidden_layer(image_features),
             self.text_features_hidden_layer(text_features)), dim=1)
        # print("concat_output_before_tanh shape: ", concat_output_before_tanh.shape)

        # in 512*bs and out 256 * bs
        z_layer_output = self.softmax_layer(
            self.z_layer(concat_output_before_tanh))
        # print("z_layer_output shape: ", z_layer_output.shape)

        # z_images will be 256 * bs
        z_images = z_layer_output * image_feats_after_tanh
        # print("z_images shape: ", z_images.shape)

        # z_texts will be 256 * bs
        z_texts = (1 - z_layer_output) * text_feats_after_tanh
        # print("z_texts shape: ", z_texts.shape)

        gate_output = z_images + z_texts
        # print("gate_output shape: ", gate_output.shape)

        after_dropout = self.drop(gate_output)

        final_output = self.fc_layer_gated(after_dropout)
        # print("final_output shape: ", final_output.shape)

        return final_output

    def get_tokenizer(self):
        return DistilBertTokenizer.from_pretrained(self.text_model_name)

    def get_image_size(self):
        return (480, 480)
    
    def get_max_token_size(self):
        return DistilBertConfig().max_position_embeddings    
