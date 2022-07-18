import utils.configuration as Conf
from utils.logger import Logger, Tracker
import utils.process_samples as Process
import utils.lr_update as lr_update
import models.models as models
import numpy as np
import data
import torch
import time
import math

# TODO: remove loss_k from config
# Load configuration parameters
parser = Conf.parser
config = Conf.Config()
args = parser.parse_args()
config.update_options(args)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = config.base_options['max_epochs']
MULTI_GPU = config.base_options['multi_gpu']

LOG_DIR = config.base_options['log_dir']
LOG_ITER = config.base_options['log_iter']
DO_VAL_ITER = config.base_options['do_val_iter']
VAL_ITER = config.base_options['val_iter']
VAL_EPOCH = config.base_options['val_epoch']

BATCH_SIZE = config.loader_options['batch_size']

TRAIN_SET = config.train_dataset_options['dataset_name']
VAL_SET = config.val_dataset_options['dataset_name']

LOAD_PRETRAINED = config.model_options['load_pretrained']
LOAD_PATH = config.model_options['pretrained_path']
MODEL_NAME = config.model_options['model_name']
MAX_SEQ_LEN = config.model_options['dec_seq_len']

CUSTOM_SCHEDULE = config.learning_options['scheduler']['do_custom']
LEARNING_RATE = config.learning_options['learning_rate']


# Create dataset
train_dataset = getattr(data, TRAIN_SET)(**config.train_dataset_options)
train_loader = torch.utils.data.DataLoader(train_dataset, **config.loader_options)
train_size = train_dataset.__len__()

val_dataset = getattr(data, VAL_SET)(**config.val_dataset_options)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=768, num_workers=4)
val_size = val_dataset.__len__()

corpus = config.corpus
config.model_options['out_num_embeddings'] = 100
config.model_options['in_num_embeddings'] = 100

# Instantiate logger
logger = Logger(config.base_options)

# Instantiate model
model = getattr(models, MODEL_NAME)(config.model_options)
if MULTI_GPU:
    assert torch.cuda.device_count() > 1
    model = torch.nn.DataParallel(model)
if LOAD_PRETRAINED:
    model.load_state_dict(torch.load(LOAD_PATH))
model.to(DEVICE)

# Instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr = LEARNING_RATE, 
                             betas=(0.9, 0.98), 
                             eps=1e-9)


# Instantiate scheduler
if CUSTOM_SCHEDULE:
    scheduler_class = torch.optim.lr_scheduler.LambdaLR
    scheduler_func = lambda x: lr_update.transformer_lr_update(x, config.learning_options)
    lr_scheduler = scheduler_class(optimizer, lr_lambda=scheduler_func)
    
# Instatiate loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

# Log options
#logger.write(vars(config))


# Log model
logger.write(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logger.write('Trainable parameters: ' + str(params))


# Log optimizer
logger.write(optimizer)
logger.write('Initial learning rate: {0:.3e}'.format(lr_update.get_lr(optimizer)))


# Log GPU
logger.write(torch.cuda.get_device_name(DEVICE))

# Instantiate tracker
tracker_LR_4by8 = Tracker(max_epochs=MAX_EPOCHS, max_iters=math.ceil(train_size/BATCH_SIZE))
tracker_RL_4by8 = Tracker(max_epochs=MAX_EPOCHS, max_iters=math.ceil(train_size/BATCH_SIZE))
tracker_LR_8by4 = Tracker(max_epochs=MAX_EPOCHS, max_iters=math.ceil(train_size/BATCH_SIZE))
tracker_RL_8by4 = Tracker(max_epochs=MAX_EPOCHS, max_iters=math.ceil(train_size/BATCH_SIZE))
tracker_Avg = Tracker(max_epochs=MAX_EPOCHS, max_iters=math.ceil(train_size/BATCH_SIZE))
# Train
if MULTI_GPU:
    dec_masks = torch.tril(torch.ones((torch.cuda.device_count(), MAX_SEQ_LEN, MAX_SEQ_LEN),dtype=torch.long)).to(DEVICE)
else:
    dec_masks = torch.tril(torch.ones((MAX_SEQ_LEN, MAX_SEQ_LEN),dtype=torch.long)).to(DEVICE)

dec_masks.requires_grad = False

if LOAD_PRETRAINED:
    iter_step = config.load_stats['iter_step']
    start_epoch = config.load_stats['start_epoch']
    tracker_LR_4by8.epoch_stats['iter_no'] = start_epoch
    tracker_LR_4by8.best_stats['val_acc_word'] = config.load_stats['best_val_acc']
    tracker_RL_4by8.epoch_stats['iter_no'] = start_epoch
    tracker_RL_4by8.best_stats['val_acc_word'] = config.load_stats['best_val_acc']
    tracker_LR_8by4.epoch_stats['iter_no'] = start_epoch
    tracker_LR_8by4.best_stats['val_acc_word'] = config.load_stats['best_val_acc']
    tracker_RL_8by4.epoch_stats['iter_no'] = start_epoch
    tracker_RL_8by4.best_stats['val_acc_word'] = config.load_stats['best_val_acc']
    tracker_Avg.epoch_stats['iter_no'] = start_epoch
    tracker_Avg.best_stats['val_acc_word'] = config.load_stats['best_val_acc']
    if CUSTOM_SCHEDULE:
        lr_scheduler.step(iter_step) ##may remove /2
else:  
    iter_step = 1
    start_epoch = 0

for epoch in range(start_epoch, MAX_EPOCHS):
    
    for dataset_output in train_loader:
        image_4by8 = dataset_output[0]
        image_8by4 = dataset_output[1]
        count = image_4by8.size(0)                

        raw_text_LR = dataset_output[2]
        raw_text_RL = dataset_output[3]
        
        model.train()
        char_count = sum([len(ele)+1 for ele in raw_text_LR])
        
        text_LR = data.word2tensor(corpus, raw_text_LR)
        text_LR = Process.apply_padding(text_LR, MAX_SEQ_LEN).to(DEVICE)
        text_RL = data.word2tensor(corpus, raw_text_RL)
        text_RL = Process.apply_padding(text_RL, MAX_SEQ_LEN).to(DEVICE)
        
        text_gt_LR = data.word2tensor_gt(corpus, raw_text_LR)
        text_gt_LR_cal_acc = Process.apply_padding(text_gt_LR, MAX_SEQ_LEN, pad=-1).to(DEVICE)
        text_gt_LR = Process.apply_padding(text_gt_LR, MAX_SEQ_LEN).to(DEVICE)
        
        text_gt_RL = data.word2tensor_gt(corpus, raw_text_RL)
        text_gt_RL_cal_acc = Process.apply_padding(text_gt_RL, MAX_SEQ_LEN, pad=-1).to(DEVICE)
        text_gt_RL = Process.apply_padding(text_gt_RL, MAX_SEQ_LEN).to(DEVICE)        
        
        image_4by8 = image_4by8.to(DEVICE)
        image_8by4 = image_8by4.to(DEVICE)

        #output = model(image, text, dec_masks,test=False)
        output = model(image_4by8,image_8by4, text_LR, text_RL, text_LR, text_RL, dec_masks)
        im_features = output[0] 
        output_labels_LR_4by8 = output[2]
        output_labels_RL_4by8 = output[3]
        output_labels_LR_8by4 = output[4]
        output_labels_RL_8by4 = output[5]        
        # compute gradient and perform GD step
        optimizer.zero_grad()
        loss_LR_4by8 = criterion(output_labels_LR_4by8.permute(0, 2, 1), text_gt_LR) # returns sum of loss for whole batch
        loss_RL_4by8 = criterion(output_labels_RL_4by8.permute(0, 2, 1), text_gt_RL)
        loss_LR_8by4 = criterion(output_labels_LR_8by4.permute(0, 2, 1), text_gt_LR) # returns sum of loss for whole batch
        loss_RL_8by4 = criterion(output_labels_RL_8by4.permute(0, 2, 1), text_gt_RL)
        
        train_loss_LR_4by8 = loss_LR_4by8.sum().item()
        train_loss_RL_4by8 = loss_RL_4by8.sum().item()
        train_loss_LR_8by4 = loss_LR_8by4.sum().item()
        train_loss_RL_8by4 = loss_RL_8by4.sum().item()
        
        loss = loss_LR_4by8.mean() + loss_RL_4by8.mean() + loss_LR_8by4.mean() + loss_RL_8by4.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optimizer.step()
        
        if CUSTOM_SCHEDULE:
            lr_scheduler.step(iter_step) ##may remove /2
        
        train_acc_char_LR_4by8 = Process.char_level_acc(output_labels_LR_4by8, text_gt_LR_cal_acc)
        output_labels_str_LR_4by8 = torch.argmax((torch.nn.functional.softmax(output_labels_LR_4by8,dim=2)), dim=2).to('cpu').numpy()
        output_labels_str_LR_4by8 = data.tensor2word(corpus, output_labels_str_LR_4by8)
        prediction_bool_LR_4by8 = np.array(output_labels_str_LR_4by8) == np.array(raw_text_LR)
        train_acc_word_LR_4by8 = np.count_nonzero(prediction_bool_LR_4by8)
        
        train_acc_char_RL_4by8 = Process.char_level_acc(output_labels_RL_4by8, text_gt_RL_cal_acc)
        output_labels_str_RL_4by8 = torch.argmax((torch.nn.functional.softmax(output_labels_RL_4by8,dim=2)), dim=2).to('cpu').numpy()
        output_labels_str_RL_4by8 = data.tensor2word(corpus, output_labels_str_RL_4by8)
        prediction_bool_RL_4by8 = np.array(output_labels_str_RL_4by8) == np.array(raw_text_RL)
        train_acc_word_RL_4by8 = np.count_nonzero(prediction_bool_RL_4by8)
        
        train_acc_char_LR_8by4 = Process.char_level_acc(output_labels_LR_8by4, text_gt_LR_cal_acc)
        output_labels_str_LR_8by4 = torch.argmax((torch.nn.functional.softmax(output_labels_LR_8by4,dim=2)), dim=2).to('cpu').numpy()
        output_labels_str_LR_8by4 = data.tensor2word(corpus, output_labels_str_LR_8by4)
        prediction_bool_LR_8by4 = np.array(output_labels_str_LR_8by4) == np.array(raw_text_LR)
        train_acc_word_LR_8by4 = np.count_nonzero(prediction_bool_LR_8by4)
        
        train_acc_char_RL_8by4 = Process.char_level_acc(output_labels_RL_8by4, text_gt_RL_cal_acc)
        output_labels_str_RL_8by4 = torch.argmax((torch.nn.functional.softmax(output_labels_RL_8by4,dim=2)), dim=2).to('cpu').numpy()
        output_labels_str_RL_8by4 = data.tensor2word(corpus, output_labels_str_RL_8by4)
        prediction_bool_RL_8by4 = np.array(output_labels_str_RL_8by4) == np.array(raw_text_RL)
        train_acc_word_RL_8by4 = np.count_nonzero(prediction_bool_RL_8by4)
        
        current_lr = lr_update.get_lr(optimizer)
        tracker_LR_4by8.update_train(count=count, char_count=char_count, train_acc_word=train_acc_word_LR_4by8, train_acc_char=train_acc_char_LR_4by8, train_loss=train_loss_LR_4by8, current_lr = current_lr)
        tracker_RL_4by8.update_train(count=count, char_count=char_count, train_acc_word=train_acc_word_RL_4by8, train_acc_char=train_acc_char_RL_4by8, train_loss=train_loss_RL_4by8, current_lr = current_lr)
        
        tracker_LR_8by4.update_train(count=count, char_count=char_count, train_acc_word=train_acc_word_LR_8by4, train_acc_char=train_acc_char_LR_8by4, train_loss=train_loss_LR_8by4, current_lr = current_lr)
        tracker_RL_8by4.update_train(count=count, char_count=char_count, train_acc_word=train_acc_word_RL_8by4, train_acc_char=train_acc_char_RL_8by4, train_loss=train_loss_RL_8by4, current_lr = current_lr)
        
        if tracker_LR_4by8.iter_stats['iter_no'] % LOG_ITER == 0:
            logger.write('4by8:')
            logger.log(tracker_LR_4by8, 'iter_stats')
            tracker_LR_4by8.reset('iter_stats')
            logger.log(tracker_RL_4by8, 'iter_stats')
            tracker_RL_4by8.reset('iter_stats')
            
            logger.write('8by4:')
            logger.log(tracker_LR_8by4, 'iter_stats')
            tracker_LR_8by4.reset('iter_stats')
            logger.log(tracker_RL_8by4, 'iter_stats')
            tracker_RL_8by4.reset('iter_stats')            
        
        if epoch > 4:
            VAL_ITER = 500
        if DO_VAL_ITER and tracker_LR_4by8.iter_stats['iter_no'] % VAL_ITER == 0:
            val_start_time = time.time()
            val_count = 0
            val_acc_word_LR_4by8 = 0
            val_acc_word_RL_4by8 = 0
            val_acc_word_LR_8by4 = 0
            val_acc_word_RL_8by4 = 0
            prob_list_LR_4by8 = []
            prob_list_RL_4by8 = []
            prob_list_LR_8by4 = []
            prob_list_RL_8by4 = []
            pred_bool_arr_LR_4by8 = np.zeros((val_size,))
            pred_bool_arr_RL_4by8 = np.zeros((val_size,))
            pred_bool_arr_LR_8by4 = np.zeros((val_size,))
            pred_bool_arr_RL_8by4 = np.zeros((val_size,)) 
            
            with torch.no_grad():

                for val_output in val_loader:
                    image_4by8 =  val_output[0]
                    image_8by4 =  val_output[1]
                    v_bs = image_4by8.size(0)
                    
                    raw_text_LR = val_output[2]
                    raw_text_RL = val_output[3]

                    val_output = Process.word_level_acc_bidirectional_dualPatches_untie(model, image_4by8,image_8by4, raw_text_LR ,raw_text_RL, corpus, MAX_SEQ_LEN, DEVICE, MULTI_GPU) 
                    val_acc_word_LR_4by8 += val_output[0] 
                    val_acc_word_RL_4by8 += val_output[1] 
                    val_acc_word_LR_8by4 += val_output[2] 
                    val_acc_word_RL_8by4 += val_output[3] 
                    
                    output_prob_LR_4by8 = val_output[4] 
                    output_prob_RL_4by8 = val_output[5] 
                    output_prob_LR_8by4 = val_output[6] 
                    output_prob_RL_8by4 = val_output[7] 
                    
                    temp_pred_bool_arr_LR_4by8 = val_output[8]
                    temp_pred_bool_arr_RL_4by8 = val_output[9]
                    temp_pred_bool_arr_LR_8by4 = val_output[10]
                    temp_pred_bool_arr_RL_8by4 = val_output[11]
            
                    output_prob_LR_4by8, _ = torch.max(output_prob_LR_4by8, dim=2)
                    output_prob_RL_4by8, _ = torch.max(output_prob_RL_4by8, dim=2)
                    output_prob_LR_8by4, _ = torch.max(output_prob_LR_8by4, dim=2)
                    output_prob_RL_8by4, _ = torch.max(output_prob_RL_8by4, dim=2)
                    
                    pred_bool_arr_LR_4by8[val_count: val_count+v_bs] = temp_pred_bool_arr_LR_4by8
                    pred_bool_arr_RL_4by8[val_count: val_count+v_bs] = temp_pred_bool_arr_RL_4by8
                    pred_bool_arr_LR_8by4[val_count: val_count+v_bs] = temp_pred_bool_arr_LR_8by4
                    pred_bool_arr_RL_8by4[val_count: val_count+v_bs] = temp_pred_bool_arr_RL_8by4                    
                    val_count += v_bs
                    
                    for i in range(v_bs):
                        text_prob_LR_4by8 = 1
                        for j in range(len(val_output[12][i])):
                            text_prob_LR_4by8 *= output_prob_LR_4by8[i, j].item()
                        prob_list_LR_4by8.append(text_prob_LR_4by8)
                    for i in range(v_bs):
                        text_prob_RL_4by8 = 1
                        for j in range(len(val_output[13][i])):
                            text_prob_RL_4by8 *= output_prob_RL_4by8[i, j].item()
                        prob_list_RL_4by8.append(text_prob_RL_4by8)
                    for i in range(v_bs):
                        text_prob_LR_8by4 = 1
                        for j in range(len(val_output[14][i])):
                            text_prob_LR_8by4 *= output_prob_LR_8by4[i, j].item()
                        prob_list_LR_8by4.append(text_prob_LR_8by4)
                    for i in range(v_bs):
                        text_prob_RL_8by4 = 1
                        for j in range(len(val_output[15][i])):
                            text_prob_RL_8by4 *= output_prob_RL_8by4[i, j].item()
                        prob_list_RL_8by4.append(text_prob_RL_8by4)

                    
            val_elasped_time = time.time() - val_start_time
            val_acc_word_LR_4by8 = val_acc_word_LR_4by8 / val_count
            val_acc_word_RL_4by8 = val_acc_word_RL_4by8 / val_count
            val_acc_word_LR_8by4 = val_acc_word_LR_8by4 / val_count
            val_acc_word_RL_8by4 = val_acc_word_RL_8by4 / val_count
            
            prob_bi = np.zeros((len(prob_list_LR_4by8), 4))
            prob_bi[:, 0] = np.array(prob_list_LR_4by8)
            prob_bi[:, 1] = np.array(prob_list_RL_4by8)
            prob_bi[:, 2] = np.array(prob_list_LR_8by4)
            prob_bi[:, 3] = np.array(prob_list_RL_8by4)

            pred_bool_bi = np.zeros((len(pred_bool_arr_LR_4by8), 4))
            pred_bool_bi[:, 0] = pred_bool_arr_LR_4by8
            pred_bool_bi[:, 1] = pred_bool_arr_RL_4by8 
            pred_bool_bi[:, 2] = pred_bool_arr_LR_8by4 
            pred_bool_bi[:, 3] = pred_bool_arr_RL_8by4 

            arg_prob_bi = np.argmax(prob_bi, axis=1)
            max_prob_pred_bool = pred_bool_bi[np.arange(val_size), arg_prob_bi]

            val_acc_word_avg = np.sum(max_prob_pred_bool)/val_size
            
            tracker_LR_4by8.update_best_val(val_acc_word_LR_4by8, epoch)
            tracker_RL_4by8.update_best_val(val_acc_word_RL_4by8, epoch)
            tracker_LR_8by4.update_best_val(val_acc_word_LR_8by4, epoch)
            tracker_RL_8by4.update_best_val(val_acc_word_RL_8by4, epoch)
            tracker_Avg.update_best_val(val_acc_word_avg, epoch)
            
            logger.write('Epoch: {0}/{1}\t val_acc_word_LR_4by8: {2:.3f}\t best_val_acc_word_LR_4by8: {3:.3f} @ epoch {4}\t time: {5:.3f}s'.format(epoch, MAX_EPOCHS-1, val_acc_word_LR_4by8, tracker_LR_4by8.best_stats['val_acc_word'], tracker_LR_4by8.best_stats['val_epoch'], val_elasped_time))
            logger.write('Epoch: {0}/{1}\t val_acc_word_RL_4by8: {2:.3f}\t best_val_acc_word_RL_4by8: {3:.3f} @ epoch {4}'.format(epoch, MAX_EPOCHS-1, val_acc_word_RL_4by8, tracker_RL_4by8.best_stats['val_acc_word'], tracker_RL_4by8.best_stats['val_epoch']))
            logger.write('Epoch: {0}/{1}\t val_acc_word_LR_8by4: {2:.3f}\t best_val_acc_word_LR_8by4: {3:.3f} @ epoch {4}\t time: {5:.3f}s'.format(epoch, MAX_EPOCHS-1, val_acc_word_LR_8by4, tracker_LR_8by4.best_stats['val_acc_word'], tracker_LR_8by4.best_stats['val_epoch'], val_elasped_time))
            logger.write('Epoch: {0}/{1}\t val_acc_word_RL_8by4: {2:.3f}\t best_val_acc_word_RL_8by4: {3:.3f} @ epoch {4}'.format(epoch, MAX_EPOCHS-1, val_acc_word_RL_8by4, tracker_RL_8by4.best_stats['val_acc_word'], tracker_RL_8by4.best_stats['val_epoch']))
            logger.write('Epoch: {0}/{1}\t val_acc_word_avg: {2:.3f}\t best_val_acc_word_avg: {3:.3f} @ epoch {4}'.format(epoch, MAX_EPOCHS-1, val_acc_word_avg, tracker_Avg.best_stats['val_acc_word'], tracker_Avg.best_stats['val_epoch']))            
            
            if val_acc_word_LR_4by8 == tracker_LR_4by8.best_stats['val_acc_word'] and val_acc_word_LR_4by8 != 0:
                torch.save(model.state_dict(), LOG_DIR+'best_LR_4by8.pth')
                with open(LOG_DIR+'best_iter_LR_4by8.txt', 'w') as f:
                    f.write(str(iter_step))
            if val_acc_word_RL_4by8 == tracker_RL_4by8.best_stats['val_acc_word'] and val_acc_word_RL_4by8 != 0:
                torch.save(model.state_dict(), LOG_DIR+'best_RL_4by8.pth')
                with open(LOG_DIR+'best_iter_RL_4by8.txt', 'w') as f:
                    f.write(str(iter_step))
            if val_acc_word_LR_8by4 == tracker_LR_8by4.best_stats['val_acc_word'] and val_acc_word_LR_8by4 != 0:
                torch.save(model.state_dict(), LOG_DIR+'best_LR_8by4.pth')
                with open(LOG_DIR+'best_iter_LR_8by4.txt', 'w') as f:
                    f.write(str(iter_step))
            if val_acc_word_RL_8by4 == tracker_RL_8by4.best_stats['val_acc_word'] and val_acc_word_RL_8by4 != 0:
                torch.save(model.state_dict(), LOG_DIR+'best_RL_8by4.pth')
                with open(LOG_DIR+'best_iter_RL_8by4.txt', 'w') as f:
                    f.write(str(iter_step)) 
            if val_acc_word_avg == tracker_Avg.best_stats['val_acc_word'] and val_acc_word_avg != 0:
                torch.save(model.state_dict(), LOG_DIR+'best_avg.pth')
                with open(LOG_DIR+'best_iter_avg.txt', 'w') as f:
                    f.write(str(iter_step))   
        iter_step += 1
            
    torch.save(model.state_dict(), LOG_DIR+ 'epoch_'+ str(epoch)+ '_' + str(iter_step) +'.pth')
    logger.log(tracker_LR_4by8, 'epoch_stats')
    logger.log(tracker_RL_4by8, 'epoch_stats')
    logger.log(tracker_LR_8by4, 'epoch_stats')
    logger.log(tracker_RL_8by4, 'epoch_stats')
    
    tracker_LR_4by8.reset('epoch_stats')   
    tracker_RL_4by8.reset('epoch_stats')   
    tracker_LR_8by4.reset('epoch_stats')   
    tracker_RL_8by4.reset('epoch_stats')   
    tracker_Avg.reset('epoch_stats')   
    
    val_start_time = time.time()
    val_count = 0
    val_acc_word_LR_4by8 = 0
    val_acc_word_RL_4by8 = 0
    val_acc_word_LR_8by4 = 0
    val_acc_word_RL_8by4 = 0
    prob_list_LR_4by8 = []
    prob_list_RL_4by8 = []
    prob_list_LR_8by4 = []
    prob_list_RL_8by4 = []
    pred_bool_arr_LR_4by8 = np.zeros((val_size,))
    pred_bool_arr_RL_4by8 = np.zeros((val_size,))
    pred_bool_arr_LR_8by4 = np.zeros((val_size,))
    pred_bool_arr_RL_8by4 = np.zeros((val_size,)) 

    with torch.no_grad():

        for val_output in val_loader:
            image_4by8 =  val_output[0]
            image_8by4 =  val_output[1]
            v_bs = image_4by8.size(0)

            raw_text_LR = val_output[2]
            raw_text_RL = val_output[3]

            val_output = Process.word_level_acc_bidirectional_dualPatches_untie(model, image_4by8,image_8by4, raw_text_LR ,raw_text_RL, corpus, MAX_SEQ_LEN, DEVICE, MULTI_GPU) 
            val_acc_word_LR_4by8 += val_output[0] 
            val_acc_word_RL_4by8 += val_output[1] 
            val_acc_word_LR_8by4 += val_output[2] 
            val_acc_word_RL_8by4 += val_output[3] 

            output_prob_LR_4by8 = val_output[4] 
            output_prob_RL_4by8 = val_output[5] 
            output_prob_LR_8by4 = val_output[6] 
            output_prob_RL_8by4 = val_output[7] 

            temp_pred_bool_arr_LR_4by8 = val_output[8]
            temp_pred_bool_arr_RL_4by8 = val_output[9]
            temp_pred_bool_arr_LR_8by4 = val_output[10]
            temp_pred_bool_arr_RL_8by4 = val_output[11]

            output_prob_LR_4by8, _ = torch.max(output_prob_LR_4by8, dim=2)
            output_prob_RL_4by8, _ = torch.max(output_prob_RL_4by8, dim=2)
            output_prob_LR_8by4, _ = torch.max(output_prob_LR_8by4, dim=2)
            output_prob_RL_8by4, _ = torch.max(output_prob_RL_8by4, dim=2)

            pred_bool_arr_LR_4by8[val_count: val_count+v_bs] = temp_pred_bool_arr_LR_4by8
            pred_bool_arr_RL_4by8[val_count: val_count+v_bs] = temp_pred_bool_arr_RL_4by8
            pred_bool_arr_LR_8by4[val_count: val_count+v_bs] = temp_pred_bool_arr_LR_8by4
            pred_bool_arr_RL_8by4[val_count: val_count+v_bs] = temp_pred_bool_arr_RL_8by4                    
            val_count += v_bs

            for i in range(v_bs):
                text_prob_LR_4by8 = 1
                for j in range(len(val_output[12][i])):
                    text_prob_LR_4by8 *= output_prob_LR_4by8[i, j].item()
                prob_list_LR_4by8.append(text_prob_LR_4by8)
            for i in range(v_bs):
                text_prob_RL_4by8 = 1
                for j in range(len(val_output[13][i])):
                    text_prob_RL_4by8 *= output_prob_RL_4by8[i, j].item()
                prob_list_RL_4by8.append(text_prob_RL_4by8)
            for i in range(v_bs):
                text_prob_LR_8by4 = 1
                for j in range(len(val_output[14][i])):
                    text_prob_LR_8by4 *= output_prob_LR_8by4[i, j].item()
                prob_list_LR_8by4.append(text_prob_LR_8by4)
            for i in range(v_bs):
                text_prob_RL_8by4 = 1
                for j in range(len(val_output[15][i])):
                    text_prob_RL_8by4 *= output_prob_RL_8by4[i, j].item()
                prob_list_RL_8by4.append(text_prob_RL_8by4)


    val_elasped_time = time.time() - val_start_time
    val_acc_word_LR_4by8 = val_acc_word_LR_4by8 / val_count
    val_acc_word_RL_4by8 = val_acc_word_RL_4by8 / val_count
    val_acc_word_LR_8by4 = val_acc_word_LR_8by4 / val_count
    val_acc_word_RL_8by4 = val_acc_word_RL_8by4 / val_count

    prob_bi = np.zeros((len(prob_list_LR_4by8), 4))
    prob_bi[:, 0] = np.array(prob_list_LR_4by8)
    prob_bi[:, 1] = np.array(prob_list_RL_4by8)
    prob_bi[:, 2] = np.array(prob_list_LR_8by4)
    prob_bi[:, 3] = np.array(prob_list_RL_8by4)

    pred_bool_bi = np.zeros((len(pred_bool_arr_LR_4by8), 4))
    pred_bool_bi[:, 0] = pred_bool_arr_LR_4by8
    pred_bool_bi[:, 1] = pred_bool_arr_RL_4by8 
    pred_bool_bi[:, 2] = pred_bool_arr_LR_8by4 
    pred_bool_bi[:, 3] = pred_bool_arr_RL_8by4 

    arg_prob_bi = np.argmax(prob_bi, axis=1)
    max_prob_pred_bool = pred_bool_bi[np.arange(val_size), arg_prob_bi]

    val_acc_word_avg = np.sum(max_prob_pred_bool)/val_size

    tracker_LR_4by8.update_best_val(val_acc_word_LR_4by8, epoch)
    tracker_RL_4by8.update_best_val(val_acc_word_RL_4by8, epoch)
    tracker_LR_8by4.update_best_val(val_acc_word_LR_8by4, epoch)
    tracker_RL_8by4.update_best_val(val_acc_word_RL_8by4, epoch)
    tracker_Avg.update_best_val(val_acc_word_avg, epoch)

    logger.write('Epoch: {0}/{1}\t val_acc_word_LR_4by8: {2:.3f}\t best_val_acc_word_LR_4by8: {3:.3f} @ epoch {4}\t time: {5:.3f}s'.format(epoch, MAX_EPOCHS-1, val_acc_word_LR_4by8, tracker_LR_4by8.best_stats['val_acc_word'], tracker_LR_4by8.best_stats['val_epoch'], val_elasped_time))
    logger.write('Epoch: {0}/{1}\t val_acc_word_RL_4by8: {2:.3f}\t best_val_acc_word_RL_4by8: {3:.3f} @ epoch {4}'.format(epoch, MAX_EPOCHS-1, val_acc_word_RL_4by8, tracker_RL_4by8.best_stats['val_acc_word'], tracker_RL_4by8.best_stats['val_epoch']))
    logger.write('Epoch: {0}/{1}\t val_acc_word_LR_8by4: {2:.3f}\t best_val_acc_word_LR_8by4: {3:.3f} @ epoch {4}\t time: {5:.3f}s'.format(epoch, MAX_EPOCHS-1, val_acc_word_LR_8by4, tracker_LR_8by4.best_stats['val_acc_word'], tracker_LR_8by4.best_stats['val_epoch'], val_elasped_time))
    logger.write('Epoch: {0}/{1}\t val_acc_word_RL_8by4: {2:.3f}\t best_val_acc_word_RL_8by4: {3:.3f} @ epoch {4}'.format(epoch, MAX_EPOCHS-1, val_acc_word_RL_8by4, tracker_RL_8by4.best_stats['val_acc_word'], tracker_RL_8by4.best_stats['val_epoch']))
    logger.write('Epoch: {0}/{1}\t val_acc_word_avg: {2:.3f}\t best_val_acc_word_avg: {3:.3f} @ epoch {4}'.format(epoch, MAX_EPOCHS-1, val_acc_word_avg, tracker_Avg.best_stats['val_acc_word'], tracker_Avg.best_stats['val_epoch']))            

    if val_acc_word_LR_4by8 == tracker_LR_4by8.best_stats['val_acc_word'] and val_acc_word_LR_4by8 != 0:
        torch.save(model.state_dict(), LOG_DIR+'best_LR_4by8.pth')
        with open(LOG_DIR+'best_iter_LR_4by8.txt', 'w') as f:
            f.write(str(iter_step))
    if val_acc_word_RL_4by8 == tracker_RL_4by8.best_stats['val_acc_word'] and val_acc_word_RL_4by8 != 0:
        torch.save(model.state_dict(), LOG_DIR+'best_RL_4by8.pth')
        with open(LOG_DIR+'best_iter_RL_4by8.txt', 'w') as f:
            f.write(str(iter_step))
    if val_acc_word_LR_8by4 == tracker_LR_8by4.best_stats['val_acc_word'] and val_acc_word_LR_8by4 != 0:
        torch.save(model.state_dict(), LOG_DIR+'best_LR_8by4.pth')
        with open(LOG_DIR+'best_iter_LR_8by4.txt', 'w') as f:
            f.write(str(iter_step))
    if val_acc_word_RL_8by4 == tracker_RL_8by4.best_stats['val_acc_word'] and val_acc_word_RL_8by4 != 0:
        torch.save(model.state_dict(), LOG_DIR+'best_RL_8by4.pth')
        with open(LOG_DIR+'best_iter_RL_8by4.txt', 'w') as f:
            f.write(str(iter_step)) 
    if val_acc_word_avg == tracker_Avg.best_stats['val_acc_word'] and val_acc_word_avg != 0:
        torch.save(model.state_dict(), LOG_DIR+'best_avg.pth')
        with open(LOG_DIR+'best_iter_avg.txt', 'w') as f:
            f.write(str(iter_step)) 