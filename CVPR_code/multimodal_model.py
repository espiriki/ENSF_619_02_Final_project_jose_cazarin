import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, BartConfig, BartForSequenceClassification
from transformers import BartModel
from transformers import BertModel, BertConfig
from torchvision.models import *
from transformers import DistilBertTokenizer, BartTokenizer
from transformers import BertTokenizer
from random import randint
import random
from torch.nn import functional as F
import numpy as np
import sys

def decision(probability):
    return np.random.rand(1)[0] < probability

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

def bart():

    model = BartModel.from_pretrained("facebook/bart-large")

    for param in model.parameters():
        param.requires_grad = False

    return model

def bert():

    model = BertModel.from_pretrained("bert-base-uncased")

    for param in model.parameters():
        param.requires_grad = False

    return model


class EffV2MediumAndDistilbertGated(nn.Module):

    def __init__(self,
                 n_classes,
                 drop_ratio,
                 image_or_text_dropout_chance,
                 img_prob_dropout,
                 num_neurons_fc,
                 text_model_name,
                 batch_size):
        super(EffV2MediumAndDistilbertGated, self).__init__()

        self.text_model_name = text_model_name
        
        if text_model_name == "bert":     
            self.text_model = bert()
        elif text_model_name == "distilbert":     
            self.text_model = distilbert()
        elif text_model_name == "bart":     
            self.text_model = bart()
        else:
            print("Wrong text model:", text_model_name)
            sys.exit(1)
            
        self.image_model = eff_net_v2()

        self.drop = nn.Dropout(p=drop_ratio)
        self.fc_layer_neurons = num_neurons_fc

        self.image_dropout = nn.Dropout2d(p=1.0)
        self.text_dropout = nn.Dropout1d(p=1.0)
        self.image_or_text_dropout_chance = image_or_text_dropout_chance
        self.img_dropout_prob = img_prob_dropout

        # 1280 from image + 768 from text
        self.image_to_hidden_size = \
            nn.Linear(1280,
                      out_features=self.fc_layer_neurons)

        print("Text model hidden size:",self.text_model.config.hidden_size)
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

        # FC layer to classes
        self.clip_fc_layer = nn.Linear(batch_size, n_classes)
        self.batch_size = batch_size

        self.trans_conv = nn.ConvTranspose1d(
            in_channels=8, out_channels=8, kernel_size=2, stride=2, padding=0, output_padding=0)

        # 0.07 is the temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_to_MMF_hidden_size = nn.Linear(1280, out_features=128)
        self.text_to_MMF_hidden_size = nn.Linear(
            in_features=self.text_model.config.hidden_size, out_features=128)
        self.MMF_relu = nn.ReLU()

        self.W_grande_1 = torch.nn.Linear(128, 128)
        self.w1 = torch.rand(128, 16)
        self.w1_final_batch = torch.rand(128, 8)

        self.w1 = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(self.w1))
        self.w1_final_batch = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(self.w1_final_batch))

        self.MMF_output = torch.nn.Linear(128, 4)

        self.MMF_128_to_256 = torch.nn.Linear(128, 256)
        self.MMF_256_to_128 = torch.nn.Linear(256, 128)
        self.MMF_dropout = nn.Dropout(p=0.2)

        self.softmax = nn.Softmax(dim=1)

        self.text_to_output = nn.Linear(
            in_features=self.text_model.config.hidden_size, out_features=4)
        self.image_to_output = nn.Linear(1280, out_features=4)

        self.MMF_output_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))


    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("Gated forward pass")
        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)
        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        
        text_features = text_output[0][:, 0]
        image_features = self.image_model(self._images)
        print("text_features:",text_features.shape)
        print("image_features:",image_features.shape)

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
        if self.text_model_name == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.text_model_name == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        elif self.text_model_name == "bart":
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")            
        
        return self.tokenizer

    def get_image_size(self):
        return (480, 480)

    def get_max_token_size(self):
        if self.text_model_name == "bert":
            self.config = BertConfig().max_position_embeddings
        elif self.text_model_name == "distilbert":
            self.config = DistilBertConfig().max_position_embeddings
        elif self.text_model_name == "bart":
            self.config = BartConfig().max_position_embeddings            
        
        return self.config

    def drop_modalities(self, _eval, remove_image, remove_text):
        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text
        if _eval:
            # when evaluating, dropout is removed
            # so set it here again
            self.image_dropout.train()
            self.text_dropout.train()
            if remove_image:
                print("    Eval: zero image")
                self._images = self.image_dropout(self._images)
            if remove_text:
                print("    Eval: zero text")
                self._input_ids = self.text_dropout(self._input_ids)
                self._attention_mask = self.text_dropout(self._attention_mask)

            if not remove_image and not remove_text:
                print("    Eval: using both")
                pass
        # Training
        else:
            print("self.image_or_text_dropout_chance: ",
                  self.image_or_text_dropout_chance)
            if decision(self.image_or_text_dropout_chance):
                image_or_text = decision(self.img_dropout_prob)
                if image_or_text:
                    print("    Train: zeroing image\n")
                    self._images = self.image_dropout(self._images)
                else:
                    print("    Train: zeroing text\n")
                    self._input_ids = self.text_dropout(self._input_ids)
                    self._attention_mask = self.text_dropout(self._attention_mask)
            else:
                print("    Train: using both\n")


class EffV2MediumAndDistilbertClassic(EffV2MediumAndDistilbertGated):
    
    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("Classic forward")
        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text

        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]
        text_features = hidden_state[:, 0]

        image_features = self.image_model(self._images)

        image_hidden_size = self.image_to_hidden_size(image_features)
        text_hidden_size = self.text_to_hidden_size(text_features)

        image_plus_text_features = torch.cat(
            (image_hidden_size, text_hidden_size), dim=1)

        after_concat = self.concat_layer(image_plus_text_features)
        after_drop = self.drop(after_concat)
        final_output = self.fc_layer(after_drop)

        return final_output
    
    
class EffV2MediumAndDistilbertNormalized(EffV2MediumAndDistilbertGated):
    
    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("Normalized forward")
        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text

        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]
        
        text_features = hidden_state[:, 0]
        image_features = self.image_model(self._images)

        image_hidden_size = self.image_to_hidden_size(image_features)
        text_hidden_size = self.text_to_hidden_size(text_features)

        image_hidden_size = image_hidden_size / image_hidden_size.norm(dim=1, keepdim=True)
        text_hidden_size = text_hidden_size / text_hidden_size.norm(dim=1, keepdim=True)

        image_plus_text_features = torch.cat(
            (image_hidden_size, text_hidden_size), dim=1)

        after_concat = self.concat_layer(image_plus_text_features)
        after_drop = self.drop(after_concat)
        final_output = self.fc_layer(after_drop)

        return final_output    



class EffV2MediumAndDistilbertCLIP(EffV2MediumAndDistilbertGated):

    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("CLIP forward")

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text

        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]

        text_features = hidden_state[:, 0]
        image_features = self.image_model(self._images)

        image_features = self.image_to_hidden_size(image_features)
        text_features = self.text_to_hidden_size(text_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        if logits_per_image.shape[0] != self.batch_size:
            print("using max unpool")
            logits_per_image = self.trans_conv(logits_per_image)

        # print("logits_per_image after max pool: ", logits_per_image.shape)

        final_output = self.clip_fc_layer(logits_per_image)

        return final_output


class EffV2MediumAndDistilbertMMF(EffV2MediumAndDistilbertGated):

    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("MMF forward")

        self._images = _images
        self._input_ids = _input_ids
        self._attention_mask = _attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]

        text_features = hidden_state[:, 0]
        image_features = self.image_model(self._images)

        text_out = self.text_to_output(text_features)
        image_out = self.image_to_output(image_features)

        text_out = self.softmax(text_out)
        image_out = self.softmax(image_out)

        # print("text out shape", text_out.shape)
        # print("image out shape", image_out.shape)

        text_features = self.text_to_MMF_hidden_size(text_features)
        image_features = self.image_to_MMF_hidden_size(image_features)

        text_features_after_relu = self.MMF_relu(text_features)
        image_features_after_relu = self.MMF_relu(image_features)

        # print("text_features_after_relu shape", text_features_after_relu.shape)
        # print("image_features_after_relu shape",
        #       image_features_after_relu.shape)

        Jf = (text_features_after_relu + image_features_after_relu) / 2

        # print("average shape", Jf.shape)

        temp_o = self.W_grande_1(torch.squeeze(Jf, 1))
        # print('temp_o.shape: ', temp_o.shape)
        utk = torch.tanh(temp_o)
        # print('utk.shape: ', utk.shape)

        if utk.shape[0] != 16:
            softmax_input = utk @ self.w1_final_batch
        else:
            softmax_input = utk @ self.w1

        # print('softmax_input.shape: ', softmax_input.shape)
        alfa_tk = self.softmax(softmax_input)
        # print('alfa_tk.shape: ', alfa_tk.shape)
        context = alfa_tk.T @ temp_o
        # print('context.shape: ', context.shape)

        final_output = self.MMF_output(context)

        # print('final_output.shape: ', final_output.shape)

        with_self_attention = context * Jf

        x1 = self.MMF_128_to_256(with_self_attention)
        x1 = self.MMF_relu(x1)
        x2 = self.MMF_256_to_128(x1)
        x2 = self.MMF_relu(x2)
        x3 = self.MMF_dropout(x2)
        final_output = self.MMF_output(x3)
        output_softmax = self.softmax(final_output)

        print("self.MMF_output_weights:", self.MMF_output_weights)

        normalized_weights = torch.softmax(self.MMF_output_weights, dim=0)

        # print("normalized_weights:", normalized_weights)

        # print("torch.sum(normalized_weights)", torch.sum(normalized_weights))

        weighted_avg = ((image_out * normalized_weights[0]) +
                        (text_out * normalized_weights[1]) +
                        (output_softmax * normalized_weights[2]))

        # print("weighted_avg shape", weighted_avg.shape)

        return weighted_avg
