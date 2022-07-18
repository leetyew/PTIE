import argparse
import yaml
import torchvision
from torch.utils.data.dataloader import default_collate
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, required=False, help="path of config yaml file"
)
parser.add_argument(
    "--log_dir", type=str, required=False, help="directory of log folder"
)
parser.add_argument(
    "--max_epochs", type=int, required=False, help="maximum epochs to interate over"
)

def image_transform(set_name):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if 'ICDAR2013Dataset_Custom' in set_name:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    elif 'ICDAR2013' in set_name:
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32,100)),
                torchvision.transforms.ToTensor()])
    elif 'Synth' in set_name:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    return transform

def custom_collate(batch):
    # Remove None from batch (Filtered out bad cases)
    batch = list(filter(lambda x : x is not None, batch))
    try:
        return default_collate(batch)
    except:
        return None

class Config():
    def __init__(self):
        self.base_options = {
        'max_epochs': 1,
        'log_dir' : 'logs/',
        'log_bool': True}
        self.model_options = {}
        self.learning_options = {}
        self.compare_options = {}
        self.train_dataset_options = {}
        self.val_dataset_options = {'collate_fn': custom_collate}
        self.test_dataset_options = {'collate_fn': custom_collate}
        self.loader_options = {'collate_fn': custom_collate}
        with open('./data/full_corpus.pkl', 'rb') as f:
            self.corpus = pickle.load(f)

        self.load_stats = {}
        
    def update_options(self, args):
        path = args.config_path
        
        with open(path, 'r') as f:
            config_yaml = yaml.safe_load(f)
        
        for k, v in config_yaml['base_options'].items():
            self.base_options[k] = v
            
        for k, v in config_yaml['model_options'].items():
            self.model_options[k] = v
        
        if self.base_options['log_dir'] == None and 'baseline' in self.model_options['model_name']:
            m_name = self.model_options['model_name']
            m_d = self.model_options['dec_d_model']
            m_h = self.model_options['decoder_heads']
            m_l = self.model_options['decoder_layers']
            log_dir = './logs/' + m_name + ',d' + str(m_d) + ',h' + str(m_h) + ',l' + str(m_l) + '/'
            self.base_options['log_dir'] = log_dir
            
        elif self.base_options['log_dir'] == None:
            raise Exception('No auto directory for this model.')
        
        if not os.path.exists(self.base_options['log_dir']):
            os.mkdir(self.base_options['log_dir'])
            
        
        for k, v in config_yaml['learning_options'].items():
            self.learning_options[k] = v
        
        for k,v in config_yaml['train_dataset_options'].items():
            self.train_dataset_options[k] = v
            
        for k,v in config_yaml['test_dataset_options'].items():
            self.test_dataset_options[k] = v    
            
        for k,v in config_yaml['val_dataset_options'].items():
            self.val_dataset_options[k] = v    
        
        for k,v in config_yaml['loader_options'].items():
            self.loader_options[k] = v
            
        for k,v in config_yaml['compare_options'].items():
            self.compare_options[k] = v

        for k,v in config_yaml['load_stats'].items():
            self.load_stats[k] = v
        
        for arg in vars(args):
            if getattr(args, arg) != None:
                self.base_options[arg] = getattr(args, arg)
                
        self.val_dataset_options['transform'] = image_transform(self.val_dataset_options['dataset_name'])
        self.train_dataset_options['transform'] = image_transform(self.train_dataset_options['dataset_name'])
        self.test_dataset_options['transform'] = image_transform(self.test_dataset_options['dataset_name'])
        with open(self.base_options['log_dir'] + 'config.yaml', 'w') as f:
            yaml.dump(config_yaml ,f)
    
    def update_options_from_path(self, path):
        ## Not completed
        with open(path, 'r') as f:
            config_yaml = yaml.safe_load(f)
        
        for k, v in config_yaml['base_options'].items():
            self.base_options[k] = v
            
        for k, v in config_yaml['model_options'].items():
            self.model_options[k] = v
        
        if self.base_options['log_dir'] == None and 'baseline' in self.model_options['model_name']:
            m_name = self.model_options['model_name']
            m_d = self.model_options['dec_d_model']
            m_h = self.model_options['decoder_heads']
            m_l = self.model_options['decoder_layers']
            log_dir = './logs/' + m_name + ',d' + str(m_d) + ',h' + str(m_h) + ',l' + str(m_l) + '/'
            self.base_options['log_dir'] = log_dir
            
        elif self.base_options['log_dir'] == None:
            raise Exception('No auto directory for this model.')
        
        if not os.path.exists(self.base_options['log_dir']):
            os.mkdir(self.base_options['log_dir'])
            
        
        for k, v in config_yaml['learning_options'].items():
            self.learning_options[k] = v
        
        for k,v in config_yaml['train_dataset_options'].items():
            self.train_dataset_options[k] = v
        
        for k,v in config_yaml['val_dataset_options'].items():
            self.val_dataset_options[k] = v

        for k,v in config_yaml['test_dataset_options'].items():
            self.test_dataset_options[k] = v    
        
        for k,v in config_yaml['loader_options'].items():
            self.loader_options[k] = v
            
        for k,v in config_yaml['compare_options'].items():
            self.compare_options[k] = v

        for k,v in config_yaml['load_stats'].items():
            self.load_stats[k] = v
        
                
        self.val_dataset_options['transform'] = image_transform(self.val_dataset_options['dataset_name'])
        self.train_dataset_options['transform'] = image_transform(self.train_dataset_options['dataset_name'])
        self.test_dataset_options['transform'] = image_transform(self.test_dataset_options['dataset_name'])
        

