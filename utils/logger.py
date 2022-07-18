import logging
import sys
import os
import time

class Tracker_Full_Port():
    def __init__(self, max_epochs, max_iters):
        
        self.current_lr = 0
        self.epoch_stats = {'max_epochs': max_epochs-1,
                            'iter_no': 0,
                            'count': 0,
                            'char_count': 0,
                            'train_acc_word_a': 0,
                            'train_acc_word_b': 0,
                            'train_acc_word_c': 0,
                            'train_acc_char': 0,
                            'train_loss': 0,
                            'train_loss_CTC': 0,
                            'start_time': time.time()
                           }
        self.iter_stats = {'max_iters': max_iters,
                           'iter_no': 1,
                           'count': 0,
                           'char_count': 0,
                           'triplet_count': 1,
                           'train_acc_word_a': 0,
                           'train_acc_word_b': 0,
                           'train_acc_word_c': 0,
                           'train_acc_char': 0,
                           'train_loss': 0,
                           'train_loss_CTC': 0,
                           'triplet_loss':0,
                           'collapse_point':0,
                           'start_time': time.time()
                          }
        self.best_stats= {'train_acc_word_c': 0,
                          'train_epoch': 0,
                          'val_acc_word_a': 0,
                          'val_acc_word_b': 0,
                          'val_acc_word_c': 0,
                          'val_epoch': 0
                         }

    def reset(self, stats_type):
        
        if stats_type == 'epoch_stats':
            self.epoch_stats['iter_no'] += 1
            self.iter_stats['iter_no'] = 0
            self.epoch_stats['count'] = 0
            self.epoch_stats['char_count'] = 0
            self.epoch_stats['train_acc_word_a'] = 0
            self.epoch_stats['train_acc_word_b'] = 0
            self.epoch_stats['train_acc_word_c'] = 0
            self.epoch_stats['train_acc_char'] = 0
            self.epoch_stats['train_loss'] = 0
            self.epoch_stats['train_loss_CTC'] = 0
            self.epoch_stats['start_time'] = time.time()

        elif stats_type == 'iter_stats':
            self.iter_stats['count'] = 0
            self.iter_stats['triplet_count'] = 1
            self.iter_stats['char_count'] = 0
            self.iter_stats['train_acc_word_a'] = 0
            self.iter_stats['train_acc_word_b'] = 0
            self.iter_stats['train_acc_word_c'] = 0
            self.iter_stats['train_acc_char'] = 0
            self.iter_stats['train_loss'] = 0
            self.iter_stats['triplet_loss'] = 0
            self.iter_stats['train_loss_CTC'] = 0
            self.iter_stats['collapse_point'] = 0
            self.iter_stats['start_time'] = time.time()
        else:
            raise Exception("No such stats.")
    
    def update_train(self, count=0, char_count=0, train_acc_word_a=0, train_acc_word_b=0, train_acc_word_c=0, train_acc_char=0, train_loss=0, current_lr=0, triplet_count=0, triplet_loss=0, collapse_point=0, add_iter=True, train_loss_CTC=0):
        
        if train_acc_word_c > self.best_stats['train_acc_word_c']:
            self.best_stats['train_acc_word_c'] = train_acc_word_c
            self.best_stats['train_epoch'] = self.epoch_stats['iter_no']
            
        self.current_lr = current_lr
        
        self.epoch_stats['count'] += count
        self.epoch_stats['char_count'] += char_count
        self.epoch_stats['train_acc_word_a'] += train_acc_word_a
        self.epoch_stats['train_acc_word_b'] += train_acc_word_b
        self.epoch_stats['train_acc_word_c'] += train_acc_word_c
        self.epoch_stats['train_acc_char'] += train_acc_char
        self.epoch_stats['train_loss'] += train_loss
        self.epoch_stats['train_loss_CTC'] += train_loss_CTC
        
        if add_iter:
            self.iter_stats['iter_no'] += 1
        
        self.iter_stats['count'] += count
        self.iter_stats['char_count'] += char_count
        self.iter_stats['train_acc_word_a'] += train_acc_word_a
        self.iter_stats['train_acc_word_b'] += train_acc_word_b
        self.iter_stats['train_acc_word_c'] += train_acc_word_c
        self.iter_stats['train_acc_char'] += train_acc_char
        self.iter_stats['train_loss'] += train_loss
        self.iter_stats['train_loss_CTC'] += train_loss_CTC
        self.iter_stats['triplet_count'] += triplet_count
        self.iter_stats['triplet_loss'] += triplet_loss
        self.iter_stats['collapse_point'] += collapse_point
        
    def update_best_val(self, val_acc_word_a, val_acc_word_b, val_acc_word_c, epoch):
        
        if val_acc_word_a > self.best_stats['val_acc_word_a']:
            self.best_stats['val_acc_word_a'] = val_acc_word_a
            self.best_stats['val_epoch'] = epoch
        if val_acc_word_b > self.best_stats['val_acc_word_b']:
            self.best_stats['val_acc_word_b'] = val_acc_word_b
        if val_acc_word_c > self.best_stats['val_acc_word_c']:
            self.best_stats['val_acc_word_c'] = val_acc_word_c
            
class Tracker():
    def __init__(self, max_epochs, max_iters):
        
        self.current_lr = 0
        self.epoch_stats = {'max_epochs': max_epochs-1,
                            'iter_no': 0,
                            'count': 0,
                            'char_count': 0,
                            'train_acc_word': 0,
                            'train_acc_char': 0,
                            'train_loss': 0,
                            'train_loss_CTC': 0,
                            'start_time': time.time()
                           }
        self.iter_stats = {'max_iters': max_iters,
                           'iter_no': 1,
                           'count': 0,
                           'char_count': 0,
                           'triplet_count': 1,
                           'train_acc_word': 0,
                           'train_acc_char': 0,
                           'train_loss': 0,
                           'train_loss_CTC': 0,
                           'triplet_loss':0,
                           'collapse_point':0,
                           'start_time': time.time()
                          }
        self.best_stats= {'train_acc_word': 0,
                          'train_epoch': 0,
                          'val_acc_word': 0,
                          'val_epoch': 0
                         }

    def reset(self, stats_type):
        
        if stats_type == 'epoch_stats':
            self.epoch_stats['iter_no'] += 1
            self.iter_stats['iter_no'] = 0
            self.epoch_stats['count'] = 0
            self.epoch_stats['char_count'] = 0
            self.epoch_stats['train_acc_word'] = 0
            self.epoch_stats['train_acc_char'] = 0
            self.epoch_stats['train_loss'] = 0
            self.epoch_stats['train_loss_CTC'] = 0
            self.epoch_stats['start_time'] = time.time()

        elif stats_type == 'iter_stats':
            self.iter_stats['count'] = 0
            self.iter_stats['triplet_count'] = 1
            self.iter_stats['char_count'] = 0
            self.iter_stats['train_acc_word'] = 0
            self.iter_stats['train_acc_char'] = 0
            self.iter_stats['train_loss'] = 0
            self.iter_stats['triplet_loss'] = 0
            self.iter_stats['train_loss_CTC'] = 0
            self.iter_stats['collapse_point'] = 0
            self.iter_stats['start_time'] = time.time()
        else:
            raise Exception("No such stats.")
    
    def update_train(self, count=0, char_count=0, train_acc_word=0, train_acc_char=0, train_loss=0, current_lr=0, triplet_count=0, triplet_loss=0, collapse_point=0, add_iter=True, train_loss_CTC=0):
        
        if train_acc_word > self.best_stats['train_acc_word']:
            self.best_stats['train_acc_word'] = train_acc_word
            self.best_stats['train_epoch'] = self.epoch_stats['iter_no']
            
        self.current_lr = current_lr
        
        self.epoch_stats['count'] += count
        self.epoch_stats['char_count'] += char_count
        self.epoch_stats['train_acc_word'] += train_acc_word
        self.epoch_stats['train_acc_char'] += train_acc_char
        self.epoch_stats['train_loss'] += train_loss
        self.epoch_stats['train_loss_CTC'] += train_loss_CTC
        
        if add_iter:
            self.iter_stats['iter_no'] += 1
        
        self.iter_stats['count'] += count
        self.iter_stats['char_count'] += char_count
        self.iter_stats['train_acc_word'] += train_acc_word
        self.iter_stats['train_acc_char'] += train_acc_char
        self.iter_stats['train_loss'] += train_loss
        self.iter_stats['train_loss_CTC'] += train_loss_CTC
        self.iter_stats['triplet_count'] += triplet_count
        self.iter_stats['triplet_loss'] += triplet_loss
        self.iter_stats['collapse_point'] += collapse_point
        
    def update_best_val(self, val_acc_word, epoch):
        
        if val_acc_word > self.best_stats['val_acc_word']:
            self.best_stats['val_acc_word'] = val_acc_word
            self.best_stats['val_epoch'] = epoch
        
     
class Logger():
    def __init__(self, options):
        
        self.log_filename = options['log_dir'] + 'train.log'
        self.log_bool = options['log_bool']
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        
        if not os.path.exists(options['log_dir']):
            os.mkdir(options['log_dir'])
        
        # Create log file if it does not exist
        if not os.path.exists(self.log_filename):
            with open(self.log_filename, 'w'): pass
        
        # Add handlers
        channel = logging.FileHandler(filename=self.log_filename, mode="a")
        channel.setFormatter(formatter)
        self.logger.addHandler(channel)
        
        channel = logging.StreamHandler(sys.stdout)
        channel.setFormatter(formatter)
        self.logger.addHandler(channel)
        
    def write(self, x, level="info"):
        
        if self.log_bool:
            getattr(self.logger, level)(str(x))
        else:
            print(str(x) + "\n")
    
    def log(self, tracker, stats_type):
        
        if stats_type == 'iter_stats':
            level = 'debug'
        else:
            level = 'info'    
            
        epoch_no = tracker.epoch_stats['iter_no']
        max_epochs = tracker.epoch_stats['max_epochs']
        iter_no = tracker.iter_stats['iter_no']
        max_iters = tracker.iter_stats['max_iters']
        lr = tracker.current_lr
        
        stats = getattr(tracker, stats_type)
        count = stats['count']
        char_count = stats['char_count']
        train_acc_char = stats['train_acc_char'] / char_count
        train_acc_word = stats['train_acc_word'] / count
        train_loss = stats['train_loss'] / count
        elasped_time = time.time() - stats['start_time']

        self.write('Epoch: {0}/{1} ({2}/{3})\t train_acc_char: {4:.3f}\t train_acc_word: {5:.3f}\t loss: {6:.3e}\t lr: {7:.2e}\t time: {8:.2f}s'.format(epoch_no, max_epochs, iter_no, max_iters, train_acc_char, train_acc_word, train_loss, lr, elasped_time), level=level)
        
    def log_Port_Full(self, tracker, stats_type):
        
        if stats_type == 'iter_stats':
            level = 'debug'
        else:
            level = 'info'    
            
        epoch_no = tracker.epoch_stats['iter_no']
        max_epochs = tracker.epoch_stats['max_epochs']
        iter_no = tracker.iter_stats['iter_no']
        max_iters = tracker.iter_stats['max_iters']
        lr = tracker.current_lr
        
        stats = getattr(tracker, stats_type)
        count = stats['count']
        char_count = stats['char_count']
        train_acc_char = stats['train_acc_char'] / char_count
        train_acc_word_a = stats['train_acc_word_a'] / count
        train_acc_word_b = stats['train_acc_word_b'] / count
        train_acc_word_c = stats['train_acc_word_c'] / count
        train_loss = stats['train_loss'] / count
        elasped_time = time.time() - stats['start_time']

        self.write('Epoch: {0}/{1} ({2}/{3})\t train_acc_char: {4:.3f}\t train_acc_word_a: {5:.3f}\t train_acc_word_b: {6:.3f}\t train_acc_word_c: {7:.3f}\t loss: {8:.3e}\t lr: {9:.2e}\t time: {10:.2f}s'.format(epoch_no, max_epochs, iter_no, max_iters, train_acc_char, train_acc_word_a, train_acc_word_b, train_acc_word_c, train_loss, lr, elasped_time), level=level)


    def log_ctc(self, tracker, stats_type):
        
        if stats_type == 'iter_stats':
            level = 'debug'
        else:
            level = 'info'    
            
        epoch_no = tracker.epoch_stats['iter_no']
        max_epochs = tracker.epoch_stats['max_epochs']
        iter_no = tracker.iter_stats['iter_no']
        max_iters = tracker.iter_stats['max_iters']
        lr = tracker.current_lr
        
        stats = getattr(tracker, stats_type)
        count = stats['count']
        char_count = stats['char_count']
        train_acc_char = stats['train_acc_char'] / char_count
        train_acc_word = stats['train_acc_word'] / count
        train_loss_CTC = stats['train_loss_CTC'] / count
        train_loss = stats['train_loss'] / count
        elasped_time = time.time() - stats['start_time']

        self.write('Epoch: {0}/{1} ({2}/{3})\t train_acc_CTC: {4:.3f}\t train_acc_CE: {5:.3f}\t loss_CTC: {6:.3e}\t loss_CE: {7:.3e}\t lr: {8:.2e}\t time: {9:.2f}s'.format(epoch_no, max_epochs, iter_no, max_iters, train_acc_char, train_acc_word, train_loss_CTC, train_loss, lr, elasped_time), level=level)
        
    def log_multi(self, tracker, stats_type):
        
        if stats_type == 'iter_stats':
            level = 'debug'
        else:
            level = 'info'    
            
        epoch_no = tracker.epoch_stats['iter_no']
        max_epochs = tracker.epoch_stats['max_epochs']
        iter_no = tracker.iter_stats['iter_no']
        max_iters = tracker.iter_stats['max_iters']
        lr = tracker.current_lr
        
        stats = getattr(tracker, stats_type)
        count = stats['count']
        char_count = stats['char_count']
        train_acc_word = stats['train_acc_word'] / count
        train_loss = stats['train_loss'] / count
        multi_loss = stats['triplet_loss'] / stats['triplet_count']
        collapse_point = stats['collapse_point']  / stats['triplet_count']
        elasped_time = time.time() - stats['start_time']

        self.write('Epoch: {0}/{1} ({2}/{3})\t train_acc_word: {4:.3f}\t rec_loss: {5:.3e}\t multi_loss: {6:.3e} \t collaspe: {7:.3e} \t lr: {8:.2e}\t time: {9:.2f}s'.format(epoch_no, max_epochs, iter_no, max_iters, train_acc_word, train_loss, multi_loss, collapse_point, lr, elasped_time), level=level)
        
    def log_con(self, tracker, stats_type):
        
        if stats_type == 'iter_stats':
            level = 'debug'
        else:
            level = 'info'    
            
        epoch_no = tracker.epoch_stats['iter_no']
        max_epochs = tracker.epoch_stats['max_epochs']
        iter_no = tracker.iter_stats['iter_no']
        max_iters = tracker.iter_stats['max_iters']
        lr = tracker.current_lr
        
        stats = getattr(tracker, stats_type)
        count = stats['count']
        train_loss = stats['train_loss'] / count
        elasped_time = time.time() - stats['start_time']

        self.write('Epoch: {0}/{1} ({2}/{3})\t loss: {4:.3e}\t lr: {5:.2e}\t time: {6:.2f}s'.format(epoch_no, max_epochs, iter_no, max_iters, train_loss, lr, elasped_time), level=level)
        
