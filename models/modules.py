# Codes are taken or referenced from 
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Translation/Transformer/
# https://github.com/SamLynnEvans/Transformer

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import torchvision


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    #nn.init.orthogonal_(m.weight, gain=1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, tokens_seq):
        seq_len = tokens_seq.size(1)
        return Variable(self.pe[0, :seq_len, :], requires_grad=False).cuda()

    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__() 
        self.linear_1 = Linear(d_model, d_ff)
        self.linear_2 = Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.gelu(self.linear_1(x))
        x = self.linear_2(x)
        return x

class MultiHeadAttention_untie(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)


        self.out = Linear(d_model, d_model)
    
    def forward(self, q, k, v, scores_pe, mask=None):
        # k=q=v for encoder and decoder first sub-layer
        # k/q/v.size(): (batch_size, seq_len, embed_size)
        # embed_size = d_model
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = ScaledDotProductAttention_untie(q, k, v, self.d_k, scores_pe, mask)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output
    
def ScaledDotProductAttention_untie(q, k, v, d_k, scores_pe, mask=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(2*d_k)
    scores = scores + scores_pe

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -100000000)
        
    scores = F.softmax(scores, dim=-1)
        
    output = torch.matmul(scores, v)
    return output


class OrthoEmbedding_bidirectional(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        weight = torch.empty((args['dec_d_model'], args['dec_d_model']), dtype=torch.float32)
        nn.init.orthogonal_(weight, gain=1)
        weight_backup = weight.clone()
        nn.init.constant_(weight[args['padding_idx']], 0)
        nn.init.constant_(weight[args['padding_idx']+100], 0)
        nn.init.constant_(weight[args['padding_idx']+200], 0)
        self.register_buffer('weight_backup', weight_backup)
        self.register_buffer('weight', weight)
    def forward(self, x, direc):
        if direc == 'LR':
            return F.embedding(x, self.weight[100:200])
        elif direc == 'RL':
            return F.embedding(x, self.weight[200:300])
        else:
            bs = x.size(0)
            return torch.cat([F.embedding(x[:bs//2, :], self.weight[100:200]), F.embedding(x[bs//2:, :], self.weight[200:300])], dim=0)
        

    
class FeaturesEncoder_dualPatches_untie(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.d_model = args['enc_d_model']
        dropout = args['enc_dropout']
        self.max_seq_len = args['enc_seq_len']
        self.encoder_layers = args['encoder_layers']
        self.h = args['encoder_heads']
        self.d_k = self.d_model // self.h
        
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerEncoderLayer_untie(args) for i in range(self.encoder_layers)])
        
        self._4by8_pe = nn.Embedding(self.max_seq_len, self.d_model) 
        self.pe_q_linear = Linear(self.d_model, self.d_model)
        self.pe_k_linear = Linear(self.d_model, self.d_model)         
        self._8by4_pe = nn.Embedding(self.max_seq_len, self.d_model)

        
    def forward(self, x, ratio, inference=False):
        if inference:
            #FASTER DECODING
            bs = x.size(0) # bs=2
            x = x * math.sqrt(self.d_model)

            pe_onehot = torch.arange(self.max_seq_len).unsqueeze(0).repeat(bs//2, 1).cuda()

            pe_4by8 = self._4by8_pe(pe_onehot) # bs=1
            pe_8by4 = self._8by4_pe(pe_onehot) # bs=1
            pe = torch.cat([pe_4by8, pe_8by4], dim=0) # bs=2

            pe_q = self.pe_q_linear(pe)
            pe_k = self.pe_k_linear(pe)

            pe_q = pe_q.view(bs, -1, self.h, self.d_k)
            pe_k = pe_k.view(bs, -1, self.h, self.d_k)
            pe_q = pe_q.transpose(1,2)
            pe_k = pe_k.transpose(1,2)        
            scores_pe = torch.matmul(pe_q, pe_k.transpose(-2, -1)) /  math.sqrt(2*self.d_k)


            x = self.dropout(x)
            for i in range(self.encoder_layers):
                x = self.layers[i](x, scores_pe)

            return x, pe
        
        else:
            #input: bs x 26 x enc_d_model
            bs = x.size(0)
            x = x * math.sqrt(self.d_model)

            pe_onehot = torch.arange(self.max_seq_len).unsqueeze(0).repeat(bs, 1).cuda()

            if ratio == '4by8':
                pe = self._4by8_pe(pe_onehot)
            elif ratio == '8by4':
                pe = self._8by4_pe(pe_onehot) 
            else:
                assert False
            pe_q = self.pe_q_linear(pe)
            pe_k = self.pe_k_linear(pe)

            pe_q = pe_q.view(bs, -1, self.h, self.d_k)
            pe_k = pe_k.view(bs, -1, self.h, self.d_k)
            pe_q = pe_q.transpose(1,2)
            pe_k = pe_k.transpose(1,2)        
            scores_pe = torch.matmul(pe_q, pe_k.transpose(-2, -1)) /  math.sqrt(2*self.d_k)


            x = self.dropout(x)
            for i in range(self.encoder_layers):
                x = self.layers[i](x, scores_pe)

            return x, pe
        
    
class TransformerEncoderLayer_untie(nn.Module):
    """Encoder layer block.
    
    https://arxiv.org/pdf/1706.03762.pdf: We apply dropout [33] to the output of each sub-layer, 
    before it is added to the sub-layer input and normalized. In addition, we apply dropout to the 
    sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
    """
    def __init__(self, args):
        super().__init__()
        
        heads = args['encoder_heads']
        d_model = args['enc_d_model']
        norm_eps = args['norm_eps']
        dropout = args['enc_dropout']
        d_ff = args['d_ff']
        
        self.attention = MultiHeadAttention_untie(heads, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model, eps=norm_eps)
        
        self.feedForward = FeedForward(d_model, d_ff)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model, eps=norm_eps)
    
    def forward(self, x, scores_pe):
        x_residual = x
        x = self.attention(x, x, x, scores_pe)
        x = self.dropout_1(x)
        x = self.norm_1(x+x_residual)
        
        x_residual = x
        x = self.feedForward(x)
        x = self.dropout_2(x)
        x = self.norm_2(x+x_residual)
        return x    
    
class TransformerDecoderLayer_untie(nn.Module):
    """Decoder layer block.
    
    https://arxiv.org/pdf/1706.03762.pdf: We apply dropout [33] to the output of each sub-layer, 
    before it is added to the sub-layer input and normalized. In addition, we apply dropout to the 
    sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
    """
    def __init__(self, args):
        super().__init__()
        
        heads = args['decoder_heads']
        d_model = args['dec_d_model']
        d_ff = args['d_ff']
        norm_eps = args['norm_eps']
        dropout = args['dec_dropout']
        
        self.maskedAttention = MultiHeadAttention_untie(heads, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model, eps=norm_eps)
        
        self.attention = MultiHeadAttention_untie(heads, d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model, eps=norm_eps)
        
        self.feedForward = FeedForward(d_model, d_ff)
        self.dropout_3 = nn.Dropout(dropout)
        self.norm_3 = nn.LayerNorm(d_model, eps=norm_eps)
    
    def forward(self, x, encoder_out, dec_scores_pe, enc_scores_pe, decoder_padding_mask):
        x_residual = x
        x = self.maskedAttention(x, x, x, dec_scores_pe, mask=decoder_padding_mask)
        x = self.dropout_1(x)
        x = self.norm_1(x+x_residual)
        
        x_residual = x
        x = self.attention(x, encoder_out, encoder_out, enc_scores_pe)
        x = self.dropout_2(x)
        x = self.norm_2(x+x_residual)
        
        x_residual = x
        x = self.feedForward(x)
        x = self.dropout_3(x)
        x = self.norm_3(x+x_residual)
        return x

class TransformerDecoder_bidirectional_dualPatches_untie(nn.Module):
    """Transformer Decoder
    
    In the embedding layers, we multiply those weights by d_model**0.5.
    """
    def __init__(self, args, **kwargs):
        super().__init__()
        
        self.d_model = args['dec_d_model']
        self.decoder_layers = args['decoder_layers']
        self.share_out_weights = args['share_out_weights']
        self.dec_max_seq_len = args['dec_seq_len']
        self.enc_max_seq_len = args['enc_seq_len']
        dropout = args['dec_dropout']
        self.h = args['encoder_heads']
        self.d_k = self.d_model // self.h
        
        if args['share_enc_dec_weights']:
            self.embedding = kwargs['decoder_embedding']
        elif args['ortho_emb']:
            self.embedding = OrthoEmbedding_bidirectional(args)
        else:
            self.embedding = Embedding(args['in_num_embeddings'], args['dec_d_model'], args['padding_idx'])


        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        self.layers.extend([TransformerDecoderLayer_untie(args) for i in range(self.decoder_layers)])
        if not self.share_out_weights:
            self.embed_out = Linear(args['dec_d_model'], args['out_num_embeddings'])
            
        self.LR_pe = nn.Embedding(self.dec_max_seq_len, self.d_model)
        self.RL_pe = nn.Embedding(self.dec_max_seq_len, self.d_model)
        
        self.pe_q_linear = Linear(self.d_model, self.d_model)
        self.pe_k_linear = Linear(self.d_model, self.d_model)
        
        self.enc_k_linear = Linear(self.d_model, self.d_model)
        self.dec_q_linear = Linear(self.d_model, self.d_model)
        
        
    def forward(self, idx_seq, encoder_out, enc_pe, direc, decoder_padding_mask, inference=False):
        
        if inference:
            # FASTER DECODING
            x = self.embedding(idx_seq, None) #idx_seq bs=4

            bs = x.size(0)
            x = x * math.sqrt(self.d_model)
            pe_onehot = torch.arange(self.dec_max_seq_len).unsqueeze(0).repeat(bs//2, 1).cuda()
            pe_LR = self.LR_pe(pe_onehot)
            pe_RL = self.RL_pe(pe_onehot)
            pe = torch.cat([pe_LR, pe_RL], dim=0)

            pe_q = self.pe_q_linear(pe)
            pe_k = self.pe_k_linear(pe)
            dec_pe_q = self.dec_q_linear(pe)
            enc_pe_k = self.enc_k_linear(enc_pe)

            pe_q = pe_q.view(bs, -1, self.h, self.d_k)
            pe_k = pe_k.view(bs, -1, self.h, self.d_k)
            dec_pe_q = dec_pe_q.view(bs, -1, self.h, self.d_k)
            enc_pe_k = enc_pe_k.view(bs, -1, self.h, self.d_k)

            pe_q = pe_q.transpose(1,2)
            pe_k = pe_k.transpose(1,2)        
            dec_pe_q = dec_pe_q.transpose(1,2) 
            enc_pe_k = enc_pe_k.transpose(1,2)   

            dec_scores_pe = torch.matmul(pe_q, pe_k.transpose(-2, -1)) /  math.sqrt(2*self.d_k)
            enc_scores_pe = torch.matmul(dec_pe_q, enc_pe_k.transpose(-2, -1)) /  math.sqrt(2*self.d_k)

            x = self.dropout(x)
            for i in range(self.decoder_layers):
                x = self.layers[i](x, encoder_out, dec_scores_pe, enc_scores_pe , decoder_padding_mask)

            if self.share_out_weights:
                out_x = F.linear(x, self.embedding.weight[:100])
            else:
                out_x = self.embed_out(x)        
            return out_x, x
        
        else:
            x = self.embedding(idx_seq, direc)
            bs = x.size(0)
            x = x * math.sqrt(self.d_model)

            pe_onehot = torch.arange(self.dec_max_seq_len).unsqueeze(0).repeat(bs, 1).cuda()

            if direc == 'LR':
                pe = self.LR_pe(pe_onehot)
            elif direc == 'RL':
                pe = self.RL_pe(pe_onehot)            

            pe_q = self.pe_q_linear(pe)
            pe_k = self.pe_k_linear(pe)
            dec_pe_q = self.dec_q_linear(pe)
            enc_pe_k = self.enc_k_linear(enc_pe)

            pe_q = pe_q.view(bs, -1, self.h, self.d_k)
            pe_k = pe_k.view(bs, -1, self.h, self.d_k)
            dec_pe_q = dec_pe_q.view(bs, -1, self.h, self.d_k)
            enc_pe_k = enc_pe_k.view(bs, -1, self.h, self.d_k)

            pe_q = pe_q.transpose(1,2)
            pe_k = pe_k.transpose(1,2)        
            dec_pe_q = dec_pe_q.transpose(1,2) 
            enc_pe_k = enc_pe_k.transpose(1,2)   

            dec_scores_pe = torch.matmul(pe_q, pe_k.transpose(-2, -1)) /  math.sqrt(2*self.d_k)
            enc_scores_pe = torch.matmul(dec_pe_q, enc_pe_k.transpose(-2, -1)) /  math.sqrt(2*self.d_k)           

            x = self.dropout(x)
            for i in range(self.decoder_layers):
                x = self.layers[i](x, encoder_out, dec_scores_pe, enc_scores_pe , decoder_padding_mask)

            if self.share_out_weights:
                out_x = F.linear(x, self.embedding.weight[:100])
            else:
                out_x = self.embed_out(x)
            return out_x, x