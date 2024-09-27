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


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Linear(d_in, d_out_kq)
        self.W_key = nn.Linear(d_in, d_out_kq)
        self.W_value = nn.Linear(d_in, d_out_v)
        self.activation_fx = nn.Tanh()

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = self.activation_fx(queries.matmul(keys.T))

        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        context_vex = attn_weights.matmul(values)
        return context_vex


class CrossAttention(nn.Module):
    def __init__(self, d_in_x1, d_in_x2, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Linear(d_in_x1, d_out_kq)
        self.W_key = nn.Linear(d_in_x2, d_out_kq)
        self.W_value = nn.Linear(d_in_x2, d_out_v)
        self.activation_fx = nn.Tanh()

    def forward(self, x_1, x_2):
        queries_1 = self.W_query(x_1)
        keys_2 = self.W_key(x_2)
        values_2 = self.W_value(x_2)

        attn_scores = self.activation_fx(queries_1.matmul(keys_2.T))

        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        assert (attn_weights.shape[0] == attn_weights.shape[1])

        dimension = attn_weights.shape[0]
        reversed_weights = (1.0-attn_weights)/(dimension-1)

        context_vec = reversed_weights.matmul(values_2)
        return context_vec

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

        self.MMF_hidden_size = 384

        self.image_to_MMF_hidden_size = nn.Linear(
            1280, out_features=self.MMF_hidden_size)
        self.text_to_MMF_hidden_size = nn.Linear(
            in_features=self.text_model.config.hidden_size, out_features=self.MMF_hidden_size)
        self.MMF_relu = nn.ReLU()

        self.MMF_dropout_25_percent = nn.Dropout(p=0.25)

        self.softmax = nn.Softmax(dim=1)

        self.output_all_features = torch.nn.Linear(640, 4)

        self.d_in, self.d_out_kq, self.d_out_v = \
            self.MMF_hidden_size, 256, self.MMF_hidden_size

        self.self_attention_image = SelfAttention(
            1280, 64, 32)
        self.self_attention_text = SelfAttention(
            768, 64, 32)

        self.huang_image = torch.nn.Linear(32, 4)
        self.huang_text = torch.nn.Linear(32, 4)
        self.huang_both = torch.nn.Linear(64, 4)
        self.average_output = nn.Parameter(torch.tensor([1.0, 1.0]))

        self.cross_attention_1 = CrossAttention(
            256, 256, 32, 64)

        self.cross_attention_2 = CrossAttention(
            256, 256, 32, 64)

        self.text_to_output = nn.Linear(self.MMF_hidden_size, out_features=4)
        self.image_to_output = nn.Linear(self.MMF_hidden_size, out_features=4)

        self.MMF_output_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        self.text_to_128 = nn.Linear(
            in_features=self.text_model.config.hidden_size, out_features=self.MMF_hidden_size)
        self.image_to_128 = nn.Linear(1280, out_features=self.MMF_hidden_size)
        self.MMF_dropout_50_percent = nn.Dropout(p=0.5)


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
            # print("self.image_or_text_dropout_chance: ",
            #       self.image_or_text_dropout_chance)
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

    def self_attention_block(self, x):
        keys = x.matmul(self.W_key)
        queries = x.matmul(self.W_query)
        values = x.matmul(self.W_value)

        # unnormalized attention weights
        attn_scores = queries.matmul(keys.T)

        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        context_vec = attn_weights.matmul(values)
        print("output shape of self attention:", context_vec.shape)
        return context_vec

    def cross_attention_block(self, x_1, x_2):
        queries_1 = x_1.matmul(self.W_query_cross)
        keys_2 = x_2.matmul(self.W_key_cross)
        values_2 = x_2.matmul(self.W_value_cross)

        attn_scores = queries_1.matmul(keys_2.T)
        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        context_vec = attn_weights.matmul(values_2)
        print("output shape of cross attention:", context_vec.shape)
        return context_vec



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

        # Get the image and text features
        original_text_features = hidden_state[:, 0]
        original_image_features = self.image_model(self._images)

        # # Normalize
        # original_text_features = original_text_features / \
        #     original_text_features.norm(dim=1, keepdim=True)
        # original_image_features = original_image_features / \
        #     original_image_features.norm(dim=1, keepdim=True)

        # image_hidden_size = self.image_to_hidden_size(original_image_features)
        # text_hidden_size = self.text_to_hidden_size(original_text_features)

        # Do attentions
        image_self_attention = self.self_attention_image(
            original_image_features)
        text_self_attention = self.self_attention_text(
            original_text_features)

        fc_layers_image = self.huang_image(image_self_attention)
        fc_layers_text = self.huang_text(text_self_attention)

        concat_features = torch.cat(
            (text_self_attention,
             image_self_attention
             ), dim=1)

        fc_layers_both = self.huang_both(concat_features)

        # sentiment_prediction_Y_M = torch.argmax(fc_layers_both, dim=1)

        # sentiment_prediction_Y_I = torch.argmax(fc_layers_image, dim=1)
        # sentiment_prediction_Y_T = torch.argmax(fc_layers_text, dim=1)

        # print("sentiment_prediction_Y_M shape:",
        #       sentiment_prediction_Y_M.shape)
        # print("sentiment_prediction_Y_I shape:",
        #       sentiment_prediction_Y_I.shape)
        # print("sentiment_prediction_Y_T shape:",
        #       sentiment_prediction_Y_T.shape)

        # print("fc_layers_both shape:", fc_layers_both.shape)
        # print("fc_layers_image shape:", fc_layers_image.shape)
        # print("fc_layers_text shape:", fc_layers_text.shape)

        # print(fc_layers_both)
        # print(sentiment_prediction_Y_M)

        # sentiment_prediction_Y_M = fc_layers_both.gather(
        #     1, sentiment_prediction_Y_M.unsqueeze(1))

        # sentiment_prediction_Y_I = fc_layers_image.gather(
        #     1, sentiment_prediction_Y_I.unsqueeze(1))

        # sentiment_prediction_Y_T = fc_layers_text.gather(
        #     1, sentiment_prediction_Y_T.unsqueeze(1))

        weights = self.average_output

        output = \
            (fc_layers_both + fc_layers_image*weights[0] +
                fc_layers_text*weights[1]) / (1 + weights[0] + weights[1])

        print("output shape:", output.shape)

        # print("my_array = ", my_array)

        # complementary_cross_attention_T_I = self.cross_attention_1(
        #     text_hidden_size, image_hidden_size)
        # complementary_cross_attention_I_T = self.cross_attention_2(
        #     image_hidden_size, text_hidden_size)

        # # Code 1
        # # Concat attn features
        # print("Option 1")
        # concat_features = torch.cat(
        #     (complementary_cross_attention_I_T,  # 64
        #      complementary_cross_attention_T_I,  # 64
        #      image_hidden_size,  # 256
        #      text_hidden_size,  # 256
        #      ), dim=1)

        # # Code 2
        # # Concat attn features
        # print("Option 2")
        # concat_features = torch.cat(
        #     (image_self_attention,  # 64
        #      text_self_attention,  # 64
        #      image_hidden_size,  # 256
        #      text_hidden_size,  # 256
        #      ), dim=1)

        # Code 3
        # Concat attn features
        # print("Option 3")
        # input_1 = image_self_attention - complementary_cross_attention_I_T
        # input_2 = text_self_attention - complementary_cross_attention_T_I
        # concat_features = torch.cat(
        #     (input_1,  # 64
        #      input_2,  # 64
        #      image_hidden_size,  # 256
        #      text_hidden_size,  # 256
        #      ), dim=1)

        # # Dropout and final layer to output
        # x3 = self.MMF_dropout_25_percent(concat_features)
        # output = self.output_all_features(x3)
        return output
