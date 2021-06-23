# -*- coding: utf-8 -*-
import torch
import torchvision
# from sklearn.model_selection import train_test_split
from torch.utils import tensorboard
import datetime
from pathlib import Path
import sys, os
from copy import deepcopy

import timm
import torchmetrics

from utils import DummyMetric, ImageLoader, log_confusion_matrix
from data import get_augmentation_tv
from serialize import load_json, save_json
#%%
run_label = f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
DEFAULT_CONFIG = {
    'run_label': run_label,
    'data': {
        'train_path': None,
        'val_path': None,
        'test_path': None,
        'augmentations': {
            'train': [('RandomHorizontalFlip', {}), ('RandomVerticalFlip', {})],
            'val': [],
            'test': []
            }
        },
    'model': {
        'name': 'resnet34',
        'num_classes': 2,
        'in_chans': 1,
        'pretrained': True
        },
    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataloader_workers': 0, #0 for windows
        'batch_size': 12,
        'num_epochs': 5,
        'optimizer': {
            'type': 'SGD',
            'kwargs': {'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.0005}
            },
        'lr_scheduler': {
            'type': 'StepLR',
            'kwargs': {'step_size': 3, 'gamma': 0.1}
            },
        'criterion': 'cross_entropy',
        'output': {
            'logging_step': 100,
            'output_dir': None,
            'weights_path': None,
            'class_names': ['alpha', 'proton']
            }
        }
}

def prepare_config(config, paths_file, run_label=None):
    if run_label is None:
        run_label = f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    paths = load_json(paths_file)

    config = deepcopy(config)
    if sys.platform == 'win32':
        config['training']['dataloader_workers'] = 0
    config['run_label'] = run_label

    config['data']['train_path'] = paths['data']['train_path']
    config['data']['val_path'] = paths['data']['val_path']
    config['data']['test_path'] = paths['data']['test_path']

    config['training']['output']['output_dir'] = os.path.join(paths['output'], run_label)
    config['training']['output']['weights_path'] = os.path.join(config['training']['output']['output_dir'], f'{run_label}_best.pt')

    return config
#%%
class CheckpointSaver:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best = None

    def save_if_best(self, model, score):
        if self.best is None or score>self.best:
            self.best = score
            device = model.device
            torch.save(model.cpu().state_dict(), self.save_path)
            model.to(device)


#%%
def get_datasets(data_config, labels=['train', 'val', 'test']):
    datasets = {}
    for label in labels:
        if data_config['augmentations'][label] is None:
            transform = None
        else:
            transform = get_augmentation_tv(data_config['augmentations'][label])
        datasets[label] = torchvision.datasets.ImageFolder(
            data_config[f'{label}_path'],
            loader=ImageLoader(depth=16),
            transform=transform)

    return [datasets[label] for label in labels]

def get_model(model_config):
    model = timm.create_model(model_config['name'],
                              num_classes = model_config['num_classes'],
                              in_chans = model_config['in_chans'],
                              pretrained = model_config['pretrained'])
    return model

def get_optimizer_and_scheduler(model, optimizer_config, scheduler_config):
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = getattr(torch.optim, optimizer_config['type'])(params,
                                                               **optimizer_config['kwargs'])
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_config['type'])(optimizer,
                                                                            **scheduler_config['kwargs'])
    return optimizer, scheduler

def get_loss_function(loss_config):
    if loss_config == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    raise NotImplementedError()

def train_one_epoch(model, data_loader, criterion, optimizer, train_config, tb_logger, metrics, state):
    device = train_config['device']
    model.train()
    for x, target in data_loader:
        target = target.to(device)
        y = model(x.to(device))
        loss = criterion(y, target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state['training_steps'] += len(target)
        metrics['loss'](loss, len(target))
        metrics['accuracy'](y.softmax(dim=-1), target)
        metrics['confusion matrix'](y.softmax(dim=-1), target)

        if state['training_steps'] - state['last_log'] >= train_config['output']['logging_step']:
            state['last_log'] = state['training_steps']

            #logging
            tb_logger.add_scalars("loss", {"train": metrics['loss'].compute()}, state['training_steps'])
            tb_logger.add_scalars("accuracy", {"train": metrics['accuracy'].compute()}, state['training_steps'])
            log_confusion_matrix(metrics['confusion matrix'].compute().cpu().numpy(),
                     tb_logger, class_names=train_config['output'].get('class_names'),
                     num_classes = metrics['confusion matrix'].num_classes,
                     image_label='train confusion matrix', epoch=state['training_steps'])

            tb_logger.flush()

            metrics['loss'].reset()
            metrics['accuracy'].reset()
            metrics['confusion matrix'].reset()

    return state

@torch.no_grad()
def validate(model, data_loader, criterion, train_config, tb_logger, metrics, state):
    device = train_config['device']
    model.eval()
    for x, target in data_loader:
        target = target.to(device)
        y = model(x.to(device))
        loss = criterion(y, target)

        metrics['loss'](loss, len(x))
        metrics['accuracy'](y.softmax(dim=-1), target)
        metrics['confusion matrix'](y.softmax(dim=-1), target)

    #logging
    val_metrics = {'loss': metrics['loss'].compute(),  'accuracy': metrics['accuracy'].compute(),
                   'cm': metrics['confusion matrix'].compute().cpu().numpy()}

    tb_logger.add_scalars("loss",  {"val": val_metrics['loss']}, state['training_steps'])
    tb_logger.add_scalars("accuracy", {"val": val_metrics['accuracy']}, state['training_steps'])

    class_names = train_config['output'].get('class_names')
    log_confusion_matrix(val_metrics['cm'],
                         tb_logger, class_names=class_names,
                         num_classes = metrics['confusion matrix'].num_classes,
                         image_label='val confusion matrix', epoch=state['epoch'])

    tb_logger.flush()

    metrics['accuracy'].reset()
    metrics['loss'].reset()
    metrics['confusion matrix'].reset()

    return val_metrics

def get_hparams(config):
    hparams = {
        'model_name': config['model']['name'],
        'batch_size': config['training']['batch_size'],
        'num_epochs': config['training']['num_epochs'],
        'criterion': config['training']['criterion'],
        'optimizer': config['training']['optimizer']['type'],
        'optimizer_momentum': config['training']['optimizer']['kwargs'].get('momentum'),
        'optimizer_weight_decay': config['training']['optimizer']['kwargs'].get('weight_decay'),
        'lr': config['training']['optimizer']['kwargs']['lr'],
        'scheduler': config['training']['lr_scheduler']['type'],
        'scheduler_step': config['training']['lr_scheduler']['kwargs'].get('step_size'),
        'scheduler_gamma': config['training']['lr_scheduler']['kwargs'].get('gamma')
        }
    return hparams


def run_training(config):
    ds_train, ds_val, ds_test = get_datasets(config['data'])
    model = get_model(config['model'])

    optimizer, scheduler = get_optimizer_and_scheduler(model,
                                                       config['training']['optimizer'],
                                                       config['training']['lr_scheduler'])
    criterion = get_loss_function(config['training']['criterion'])

    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True,
                                           batch_size=config['training']['batch_size'],
                                           pin_memory=True,
                                           num_workers=config['training']['dataloader_workers']
                                           )
    dl_val = torch.utils.data.DataLoader(ds_val, shuffle=False,
                                       batch_size=config['training']['batch_size'],
                                       pin_memory=True,
                                       num_workers=config['training']['dataloader_workers']
                                       )
    dl_test = torch.utils.data.DataLoader(ds_test, shuffle=False,
                                       batch_size=config['training']['batch_size'],
                                       pin_memory=True,
                                       num_workers=config['training']['dataloader_workers']
                                       )

    metrics = {
        'accuracy': torchmetrics.Accuracy().to(config['training']['device']),
        'loss': DummyMetric().to(config['training']['device']),
        'confusion matrix': torchmetrics.ConfusionMatrix(
            num_classes=config['model']['num_classes'], normalize='true'
            ).to(config['training']['device'])
        }

    output_dir = Path(config['training']['output']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    save_json(config, output_dir / 'config.json')
    tb_logger = tensorboard.SummaryWriter(config['training']['output']['output_dir'])
    checkpoint_saver = CheckpointSaver(config['training']['output']['weights_path'])

    model = model.to(config['training']['device'])
    state = {'training_steps': 0, 'epoch': None, 'last_log': -float('inf')}
    for i_epoch in range(config['training']['num_epochs']):
        state['epoch'] = i_epoch
        state = train_one_epoch(model, dl_train, criterion, optimizer, config['training'], tb_logger, metrics, state)
        val_metrics = validate(model, dl_val, criterion, config['training'], tb_logger, metrics, state)

        checkpoint_saver.save_if_best(model, val_metrics['accuracy'])
        # weights_file = f'weights_{i_epoch}_acc_{val_metrics["accuracy"]*100:0.2f}.pt'
        # torch.save(model, output_dir / weights_file)
    test_metrics = validate(model, dl_test, criterion, config['training'], tb_logger, metrics, state)
    metrics_out = {'val_accuracy': val_metrics['accuracy'], 'val_loss': val_metrics['loss'],
                           'test_accuracy': test_metrics['accuracy'], 'test_loss': test_metrics['loss']
                           }
    tb_logger.add_hparams(get_hparams(config), metrics_out)
    tb_logger.close()

    return metrics_out

if __name__ == '__main__':
    paths_file = 'paths.json'
    config = prepare_config(DEFAULT_CONFIG, paths_file)
    run_training(config)


