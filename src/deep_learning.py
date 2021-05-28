# -*- coding: utf-8 -*-
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils import tensorboard
import datetime
from pathlib import Path
import os

import timm
import torchmetrics

from utils import DummyMetric, ImageLoader, log_confusion_matrix

run_label = f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
default_config = {
    'run_label': run_label,
    'data': {
        'path': r'../data/labeled_data_210513',
        'val_size': 0.2
        },
    'model': {
        'name': 'resnet34',
        'num_classes': 3,
        'in_chans': 1,
        'pretrained': True
        },
    'training': {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
            'output_dir': rf'../output/logs/{run_label}',
            'class_names': ['alpha_decay', 'detector_effects', 'natural_bckg']
            }
        }

}

def get_datasets(data_config):
    transform = None
    # torchvision.transforms.Compose([
        # torchvision.transforms.Resize((224,224)),
        # torchvision.transforms.ToTensor()
        # ])
    ds_all = torchvision.datasets.ImageFolder(data_config['path'], transform=transform, loader=ImageLoader(depth=16))
    train_indices, test_indices = train_test_split(list(range(len(ds_all))), test_size=data_config['val_size'])
    ds_train = torch.utils.data.Subset(ds_all, train_indices)
    ds_test = torch.utils.data.Subset(ds_all, test_indices)
    return ds_train, ds_test

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
        metrics['loss'](loss.detach(), len(target))
        metrics['accuracy'](y.softmax(dim=-1), target)

        if state['training_steps'] - state['last_log'] >= train_config['output']['logging_step']:
            state['last_log'] = state['training_steps']

            #logging
            tb_logger.add_scalars("loss", {"train": metrics['loss'].compute()}, state['training_steps'])
            tb_logger.add_scalars("accuracy", {"train": metrics['accuracy'].compute()}, state['training_steps'])
            tb_logger.flush()

            metrics['loss'].reset()
            metrics['accuracy'].reset()

    return state

@torch.no_grad()
def validate(model, data_loader, criterion, train_config, tb_logger, metrics, state):
    device = train_config['device']
    model.eval()
    data_storage = {'true_labels': [], 'pred_labels': []}
    for x, target in data_loader:
        target = target.to(device)
        y = model(x.to(device))
        loss = criterion(y, target)

        metrics['loss'](loss, len(x))
        metrics['accuracy'](y.softmax(dim=-1), target)

        data_storage['true_labels'] += list(target.cpu().tolist())
        data_storage['pred_labels'] += list(y.cpu().argmax(1).tolist())

    #logging
    val_metrics = {'loss': metrics['loss'].compute(),  'acc': metrics['accuracy'].compute()}

    tb_logger.add_scalars("loss",  {"val": val_metrics['loss']}, state['training_steps'])
    tb_logger.add_scalars("accuracy", {"val": val_metrics['acc']}, state['training_steps'])

    class_names = train_config['output'].get('class_names')
    log_confusion_matrix(data_storage['true_labels'], data_storage['pred_labels'],
                         tb_logger, class_names=class_names,
                         image_label='confusion matrix', epoch=state['epoch'])

    tb_logger.flush()

    metrics['accuracy'].reset()
    metrics['loss'].reset()

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
    ds_train, ds_val = get_datasets(config['data'])
    model = get_model(config['model'])

    optimizer, scheduler = get_optimizer_and_scheduler(model,
                                                       config['training']['optimizer'],
                                                       config['training']['lr_scheduler'])
    criterion = get_loss_function(config['training']['criterion'])

    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True,
                                           batch_size=config['training']['batch_size'],
                                           pin_memory=False
                                           )
    dl_val = torch.utils.data.DataLoader(ds_train, shuffle=False,
                                       batch_size=config['training']['batch_size'],
                                       pin_memory=False
                                       )

    metrics = {
        'accuracy': torchmetrics.Accuracy().to(config['training']['device']),
        'loss': DummyMetric().to(config['training']['device'])
        }

    output_dir = Path(config['training']['output']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    tb_logger = tensorboard.SummaryWriter(config['training']['output']['output_dir'])

    model = model.to(config['training']['device'])
    state = {'training_steps': 0, 'epoch': None, 'last_log': -float('inf')}
    for i_epoch in range(config['training']['num_epochs']):
        state['epoch'] = i_epoch
        state = train_one_epoch(model, dl_train, criterion, optimizer, config['training'], tb_logger, metrics, state)
        val_metrics = validate(model, dl_val, criterion, config['training'], tb_logger, metrics, state)

        weights_file = f'weights_{i_epoch}_acc_{val_metrics["acc"]*100:0.2f}.pt'
        torch.save(model, output_dir / weights_file)

    tb_logger.add_hparams(get_hparams(config), val_metrics)
    tb_logger.close()

if __name__ == '__main__':
    run_training(default_config)


