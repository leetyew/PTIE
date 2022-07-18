def transformer_lr_update(iter_step, lr_config):
    warmup = lr_config['scheduler']['warmup_steps']
    if iter_step == 0:
        return min(1*1**-0.5, 1*warmup**-1.5)
    else:
        return min(1*iter_step**-0.5, iter_step*warmup**-1.5)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def decay_lr_update(iter_step, lr_config):
    #lr = lr_config['learning_rate']
    return 0.9**(iter_step)

def STN_lr_update(iter_step, lr_config):
    warmup = lr_config['scheduler']['warmup_steps']
    if iter_step == 0:
        return min(1*1**-0.80, 1*warmup**-1.80)
    else:
        return min(1*iter_step**-0.80, iter_step*warmup**-1.80)