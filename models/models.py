import torch
from torch import nn
import torch.nn.functional as F
import models.modules as modules
import time

    
class ViT_bidirectional_dualPatches_untie(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc_d_model = int(args['enc_d_model'])
        self.linear_4by8 = modules.Linear(32, self.enc_d_model)
        self.linear_8by4 = modules.Linear(32, self.enc_d_model)
        self.image_enc = modules.FeaturesEncoder_dualPatches_untie(args)
        self.image_dec = modules.TransformerDecoder_bidirectional_dualPatches_untie(args)

        
    def forward(self, image_4by8, image_8by4, idx_seq_LR_4by8, idx_seq_RL_4by8, idx_seq_LR_8by4, idx_seq_RL_8by4, decoder_padding_mask=None, im_features_4by8=None, im_features_8by4=None, enc_pe_4by8=None, enc_pe_8by4=None):

        bs = image_4by8.size(0)
        if torch.is_tensor(im_features_4by8) and torch.is_tensor(im_features_8by4):
            im_seq_features_4by8 = im_features_4by8
            im_seq_features_8by4 = im_features_8by4
        else:
            im_out = self.linear_4by8(image_4by8) #output of bs x enc_d_model x 8 x 25 (if data used is 32 by 100)
            encoder_out = self.image_enc(im_out, '4by8') #output of bs x enc_seq_len x enc_d_model
            im_seq_features_4by8 = encoder_out[0]
            enc_pe_4by8 = encoder_out[1]
            
            
            im_out = self.linear_8by4(image_8by4) #output of bs x enc_d_model x 8 x 25 (if data used is 32 by 100)
            encoder_out = self.image_enc(im_out, '8by4') #output of bs x enc_seq_len x enc_d_model
            im_seq_features_8by4 = encoder_out[0]
            enc_pe_8by4 = encoder_out[1]    
            
        output_LR_4by8, embed_LR_4by8 = self.image_dec(idx_seq_LR_4by8, im_seq_features_4by8, enc_pe_4by8, 'LR', decoder_padding_mask)
        output_RL_4by8, embed_RL_4by8 = self.image_dec(idx_seq_RL_4by8, im_seq_features_4by8, enc_pe_4by8, 'RL', decoder_padding_mask)
        
        output_LR_8by4, embed_LR_8by4 = self.image_dec(idx_seq_LR_8by4, im_seq_features_8by4, enc_pe_8by4, 'LR', decoder_padding_mask)
        output_RL_8by4, embed_RL_8by4 = self.image_dec(idx_seq_RL_8by4, im_seq_features_8by4, enc_pe_8by4, 'RL', decoder_padding_mask)        
        return im_seq_features_4by8, im_seq_features_8by4, output_LR_4by8, output_RL_4by8, output_LR_8by4, output_RL_8by4, enc_pe_4by8, enc_pe_8by4

    
class ViT_bidirectional_dualPatches_untie_inference(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.enc_d_model = int(args['enc_d_model'])
        self.linear_4by8 = modules.Linear(32, self.enc_d_model)
        self.linear_8by4 = modules.Linear(32, self.enc_d_model)
        self.image_enc = modules.FeaturesEncoder_dualPatches_untie(args)
        self.image_dec = modules.TransformerDecoder_bidirectional_dualPatches_untie(args)

        
    # FASTER DECODING
    def forward(self, image_4by8, image_8by4, idx_seq_, decoder_padding_mask=None, im_features_=None, enc_pe_=None):

        bs = image_4by8.size(0)
        if torch.is_tensor(im_features_):
            im_seq_features = im_features_ #bs=4
        else:
            im_out_4by8 = self.linear_4by8(image_4by8) #bs=1
            im_out_8by4 = self.linear_8by4(image_8by4) #bs=1

            im_out = torch.cat([im_out_4by8, im_out_8by4]) #bs=2
            encoder_out = self.image_enc(im_out, '4by8', inference=True) #bs=2
            im_seq_features = torch.cat([encoder_out[0], encoder_out[0]], dim=0) #bs=4
            enc_pe_ = torch.cat([encoder_out[1], encoder_out[1]], dim=0) #bs=4
            
        output, embed_ = self.image_dec(idx_seq_, im_seq_features, enc_pe_, None, decoder_padding_mask, inference=True)
      
        return im_seq_features, output, enc_pe_, embed_
    
    
