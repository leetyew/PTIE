import numpy as np
import torch
import data
import math

def process_output(output, label_idx):
    batch_size = output.size(0)
    corpus_size = output.size(2)
    temp_out = torch.zeros([batch_size, corpus_size]).cuda()
    
    for i in range(len(label_idx)):
        temp_out[i, :] = output[i, label_idx[i], :]
    
    return temp_out
    
    
def apply_padding(seq, max_seq_len, pad=0):
    batch_size = len(seq)
    seq_out = torch.ones([batch_size, max_seq_len], dtype=torch.long) * pad
    for i in range(batch_size):
        seq_out[i, :len(seq[i])] = seq[i]
    
    return seq_out

def char_level_acc(output_labels, text_gt_cal_acc):
    # output_labels in batch_size x seq_len x corpus_size
    # text_gt_cal_acc in batch_size x seq_len
    _, output_labels = torch.max(output_labels, -1)
    acc = torch.sum(torch.sum(torch.eq(output_labels, text_gt_cal_acc), dim=1)).item()
    
    return acc


def word_level_acc_bidirectional_dualPatches_untie(model, image_4by8, image_8by4, text_LR, text_RL, corpus, max_seq_len, device, multi_gpu, single_case=True, with_sym=False, im_features=None):
    alphabets_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabets_lower = alphabets_upper.lower()
    numerals = '0123456789'
    alphanumeric = list(alphabets_upper) + list(alphabets_lower) + list(numerals)
    alphanumeric = list(set(alphanumeric))    
        
    model.eval()
    bs = image_4by8.size(0)
    acc = 0.
    text_LR = list(text_LR)
    text_RL = list(text_RL)
    
    for i in range(len(text_LR)):
        new_text = ''
        for c in text_LR[i]:
            if c in alphanumeric:
                new_text += c
        text_LR[i] = new_text
        
    for i in range(len(text_RL)):
        new_text = ''
        for c in text_RL[i]:
            if c in alphanumeric:
                new_text += c
        text_RL[i] = new_text
    
    if multi_gpu:
        dec_masks = torch.tril(torch.ones((torch.cuda.device_count(), max_seq_len, max_seq_len),dtype=torch.long)).to(device)
    else:
        dec_masks = torch.tril(torch.ones((max_seq_len, max_seq_len),dtype=torch.long)).to(device)
    dec_masks.requires_grad=False
    
    image_4by8 = image_4by8.to(device)
    image_8by4 = image_8by4.to(device)
    
    input_seq_LR_4by8 = torch.zeros([bs, max_seq_len], dtype=torch.long).to(device)
    input_seq_LR_4by8[:, 0] = torch.ones([bs], dtype=torch.long)
    input_seq_RL_4by8 = torch.zeros([bs, max_seq_len], dtype=torch.long).to(device)
    input_seq_RL_4by8[:, 0] = torch.ones([bs], dtype=torch.long)
    
    input_seq_LR_8by4 = torch.zeros([bs, max_seq_len], dtype=torch.long).to(device)
    input_seq_LR_8by4[:, 0] = torch.ones([bs], dtype=torch.long)
    input_seq_RL_8by4 = torch.zeros([bs, max_seq_len], dtype=torch.long).to(device)
    input_seq_RL_8by4[:, 0] = torch.ones([bs], dtype=torch.long)    
    gt_len = torch.tensor([len(ele)+2 for ele in text_LR], dtype=torch.long).to(device)
    max_gt_len = torch.max(gt_len).item()

    idx = 1
        
    eow = False #End of word
    while not eow:
        if idx == 1 and not torch.is_tensor(im_features):
            output = model(image_4by8, image_8by4, input_seq_LR_4by8, input_seq_RL_4by8, input_seq_LR_8by4, input_seq_RL_8by4, dec_masks)
            #output = model(image, input_seq, dec_masks)
            im_features_4by8 = output[0]
            im_features_8by4 = output[1]
            output_label_LR_4by8 = output[2]
            output_label_RL_4by8 = output[3]
            output_label_LR_8by4 = output[4]
            output_label_RL_8by4 = output[5]
            enc_pe_4by8 = output[6]
            enc_pe_8by4 = output[7]
        else:
            output = model(image_4by8, image_8by4, input_seq_LR_4by8, input_seq_RL_4by8, input_seq_LR_8by4, input_seq_RL_8by4, dec_masks, im_features_4by8=im_features_4by8, im_features_8by4=im_features_8by4,  enc_pe_4by8=enc_pe_4by8, enc_pe_8by4=enc_pe_8by4)

        output_label_LR_4by8 = output[2]
        output_label_LR_4by8 = output_label_LR_4by8[:, idx-1, :]
        _, output_index_LR_4by8 = torch.max(output_label_LR_4by8, dim=-1)
        input_seq_LR_4by8[:, idx] = output_index_LR_4by8
        
        output_label_RL_4by8 = output[3]
        output_label_RL_4by8 = output_label_RL_4by8[:, idx-1, :]
        _, output_index_RL_4by8 = torch.max(output_label_RL_4by8, dim=-1)
        input_seq_RL_4by8[:, idx] = output_index_RL_4by8
        
        output_label_LR_8by4 = output[4]
        output_label_LR_8by4 = output_label_LR_8by4[:, idx-1, :]
        _, output_index_LR_8by4 = torch.max(output_label_LR_8by4, dim=-1)
        input_seq_LR_8by4[:, idx] = output_index_LR_8by4
        
        output_label_RL_8by4 = output[5]
        output_label_RL_8by4 = output_label_RL_8by4[:, idx-1, :]
        _, output_index_RL_8by4 = torch.max(output_label_RL_8by4, dim=-1)
        input_seq_RL_8by4[:, idx] = output_index_RL_8by4      
        idx += 1

        if idx == max_gt_len:
            eow = True
    
    single_case = True
    with_sym = False  
    
    if single_case:
        for i in range(len(text_LR)):
            text_LR[i] = text_LR[i].lower()
            text_RL[i] = text_RL[i].lower()
    
    raw_input_seq_str_LR_4by8 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_LR_4by8[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_LR_4by8[i][j].item()]
                temp_str += next_char.lower()
                
        raw_input_seq_str_LR_4by8.append(temp_str)
        
    raw_input_seq_str_RL_4by8 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_RL_4by8[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_RL_4by8[i][j].item()]
                temp_str += next_char.lower()                    

        raw_input_seq_str_RL_4by8.append(temp_str)

    raw_input_seq_str_LR_8by4 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_LR_8by4[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_LR_8by4[i][j].item()]
                temp_str += next_char.lower()                    

        raw_input_seq_str_LR_8by4.append(temp_str)
        
    raw_input_seq_str_RL_8by4 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_RL_8by4[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_RL_8by4[i][j].item()]
                temp_str += next_char.lower()                    

        raw_input_seq_str_RL_8by4.append(temp_str)
    
    input_seq_str_LR_4by8 = []
    for i in range(len(raw_input_seq_str_LR_4by8)):
        new_text = ''
        for c in raw_input_seq_str_LR_4by8[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_LR_4by8.append(new_text)
        
    input_seq_str_RL_4by8 = []
    for i in range(len(raw_input_seq_str_RL_4by8)):
        new_text = ''
        for c in raw_input_seq_str_RL_4by8[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_RL_4by8.append(new_text)
        
    input_seq_str_LR_8by4 = []
    for i in range(len(raw_input_seq_str_LR_8by4)):
        new_text = ''
        for c in raw_input_seq_str_LR_8by4[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_LR_8by4.append(new_text)
        
    input_seq_str_RL_8by4 = []
    for i in range(len(raw_input_seq_str_RL_8by4)):
        new_text = ''
        for c in raw_input_seq_str_RL_8by4[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_RL_8by4.append(new_text)
        
    prediction_bool_LR_4by8 = np.array(input_seq_str_LR_4by8) == np.array(text_LR)
    acc_LR_4by8 = np.count_nonzero(prediction_bool_LR_4by8)
                        
    prediction_bool_RL_4by8 = np.array(input_seq_str_RL_4by8) == np.array(text_RL)
    acc_RL_4by8 = np.count_nonzero(prediction_bool_RL_4by8)
    
    prediction_bool_LR_8by4 = np.array(input_seq_str_LR_8by4) == np.array(text_LR)
    acc_LR_8by4 = np.count_nonzero(prediction_bool_LR_8by4)
                        
    prediction_bool_RL_8by4 = np.array(input_seq_str_RL_8by4) == np.array(text_RL)
    acc_RL_8by4 = np.count_nonzero(prediction_bool_RL_8by4)
    
    softmax = torch.nn.Softmax(dim=2)
    output_prob_LR_4by8 = softmax(output[2])
    output_prob_RL_4by8 = softmax(output[3])
    output_prob_LR_8by4 = softmax(output[4])
    output_prob_RL_8by4 = softmax(output[5])
    
    return acc_LR_4by8, acc_RL_4by8, acc_LR_8by4, acc_RL_8by4, output_prob_LR_4by8, output_prob_RL_4by8, output_prob_LR_8by4, output_prob_RL_8by4, prediction_bool_LR_4by8, prediction_bool_RL_4by8, prediction_bool_LR_8by4, prediction_bool_RL_8by4, raw_input_seq_str_LR_4by8, raw_input_seq_str_RL_4by8, raw_input_seq_str_LR_8by4, raw_input_seq_str_RL_8by4

def word_level_acc_bidirectional_dualPatches_untie_inference(model, image_4by8, image_8by4, text_LR, text_RL, corpus, max_seq_len, device, multi_gpu, single_case=True, with_sym=False, im_features=None):
    alphabets_upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabets_lower = alphabets_upper.lower()
    numerals = '0123456789'
    alphanumeric = list(alphabets_upper) + list(alphabets_lower) + list(numerals)
    alphanumeric = list(set(alphanumeric))    
    softmax = torch.nn.Softmax(dim=2)
    model.eval()
    bs = image_4by8.size(0)
    acc = 0.
    text_LR = list(text_LR)
    text_RL = list(text_RL)
    
    for i in range(len(text_LR)):
        new_text = ''
        for c in text_LR[i]:
            if c in alphanumeric:
                new_text += c
        text_LR[i] = new_text
        
    for i in range(len(text_RL)):
        new_text = ''
        for c in text_RL[i]:
            if c in alphanumeric:
                new_text += c
        text_RL[i] = new_text
        
    if multi_gpu:
        dec_masks = torch.tril(torch.ones((torch.cuda.device_count(), max_seq_len, max_seq_len),dtype=torch.long)).to(device)
    else:
        dec_masks = torch.tril(torch.ones((max_seq_len, max_seq_len),dtype=torch.long)).to(device)
    dec_masks.requires_grad=False
    
    image_4by8 = image_4by8.to(device)
    image_8by4 = image_8by4.to(device)
    
    input_seq_ = torch.zeros([bs*4, max_seq_len], dtype=torch.long).to(device)
    input_seq_[:, 0] = torch.ones([bs*4], dtype=torch.long)

    gt_len = torch.tensor([len(ele)+2 for ele in text_LR], dtype=torch.long).to(device)
    max_gt_len = torch.max(gt_len).item()

    idx = 1
        
    eow = False #End of word
    
    while not eow:
        if idx == 1 and not torch.is_tensor(im_features):
            output = model(image_4by8, image_8by4, input_seq_, dec_masks)
            im_features_ = output[0]
            output_label_ = output[1]
            enc_pe_ = output[2]
        else:
            output = model(image_4by8, image_8by4, input_seq_, dec_masks, im_features_=im_features_, enc_pe_=enc_pe_)
            output_label_ = output[1]

        output_label_ = output_label_[:, idx-1, :]
        _, output_index_ = torch.max(output_label_, dim=-1)
        input_seq_[:, idx] = output_index_
     
        idx += 1

        if idx == max_gt_len:
            eow = True
            
    input_seq_LR_4by8 = input_seq_[:bs]
    input_seq_LR_8by4 = input_seq_[bs:bs*2]
    input_seq_RL_4by8 = input_seq_[bs*2:bs*3]
    input_seq_RL_8by4 = input_seq_[bs*3:]
    
    single_case = True
    with_sym = False  
    
    if single_case:
        for i in range(len(text_LR)):
            text_LR[i] = text_LR[i].lower()
            text_RL[i] = text_RL[i].lower()
    
    raw_input_seq_str_LR_4by8 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_LR_4by8[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_LR_4by8[i][j].item()]
                temp_str += next_char.lower()
                
        raw_input_seq_str_LR_4by8.append(temp_str)
        
    raw_input_seq_str_RL_4by8 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_RL_4by8[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_RL_4by8[i][j].item()]
                temp_str += next_char.lower()                    

        raw_input_seq_str_RL_4by8.append(temp_str)

    raw_input_seq_str_LR_8by4 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_LR_8by4[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_LR_8by4[i][j].item()]
                temp_str += next_char.lower()                    

        raw_input_seq_str_LR_8by4.append(temp_str)
        
    raw_input_seq_str_RL_8by4 = []
    for i in range(bs):
        temp_str = ''
        for j in range(1, max_seq_len):
            if input_seq_RL_8by4[i][j].item() == 2:
                break
            else:
                next_char = corpus['idx2char'][input_seq_RL_8by4[i][j].item()]
                temp_str += next_char.lower()                    

        raw_input_seq_str_RL_8by4.append(temp_str)
    
    input_seq_str_LR_4by8 = []
    for i in range(len(raw_input_seq_str_LR_4by8)):
        new_text = ''
        for c in raw_input_seq_str_LR_4by8[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_LR_4by8.append(new_text)
        
    input_seq_str_RL_4by8 = []
    for i in range(len(raw_input_seq_str_RL_4by8)):
        new_text = ''
        for c in raw_input_seq_str_RL_4by8[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_RL_4by8.append(new_text)
        
    input_seq_str_LR_8by4 = []
    for i in range(len(raw_input_seq_str_LR_8by4)):
        new_text = ''
        for c in raw_input_seq_str_LR_8by4[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_LR_8by4.append(new_text)
        
    input_seq_str_RL_8by4 = []
    for i in range(len(raw_input_seq_str_RL_8by4)):
        new_text = ''
        for c in raw_input_seq_str_RL_8by4[i]:
            if c in alphanumeric:
                new_text += c
        input_seq_str_RL_8by4.append(new_text)
        
    prediction_bool_LR_4by8 = np.array(input_seq_str_LR_4by8) == np.array(text_LR)
    acc_LR_4by8 = np.count_nonzero(prediction_bool_LR_4by8)
                        
    prediction_bool_RL_4by8 = np.array(input_seq_str_RL_4by8) == np.array(text_RL)
    acc_RL_4by8 = np.count_nonzero(prediction_bool_RL_4by8)
    
    prediction_bool_LR_8by4 = np.array(input_seq_str_LR_8by4) == np.array(text_LR)
    acc_LR_8by4 = np.count_nonzero(prediction_bool_LR_8by4)
                        
    prediction_bool_RL_8by4 = np.array(input_seq_str_RL_8by4) == np.array(text_RL)
    acc_RL_8by4 = np.count_nonzero(prediction_bool_RL_8by4)

    output_prob_LR_4by8 = softmax(output[1][:bs, :, :])
    output_prob_LR_8by4 = softmax(output[1][bs:bs*2, :, :])
    output_prob_RL_4by8 = softmax(output[1][bs*2:bs*3, :, :])
    output_prob_RL_8by4 = softmax(output[1][bs*3:, :, :])

    return acc_LR_4by8, acc_RL_4by8, acc_LR_8by4, acc_RL_8by4, prediction_bool_LR_4by8, prediction_bool_RL_4by8, prediction_bool_LR_8by4, prediction_bool_RL_8by4, text_LR, text_RL, input_seq_str_LR_4by8, input_seq_str_RL_4by8, input_seq_str_LR_8by4, input_seq_str_RL_8by4, output_prob_LR_4by8, output_prob_RL_4by8, output_prob_LR_8by4, output_prob_RL_8by4 