# -*- coding: utf-8 -*-

from hparams_tune import hoptim_loop, HOptimConfig

config = HOptimConfig()
config.train_config['training']['num_epochs'] = 5
config.train_config['training']['dataloader_workers'] = 8
config.train_config['training']['batch_size'] = 30

config.params = [
        {'type': 'Choice', 'kwargs': {'name': 'model_name', 'range': ['resnet18', 'resnet34', 'resnet50', 'densenet121', 'efficientnet_b0']}},
        {'type': 'Choice', 'kwargs': {'name': 'model_pretrained', 'range': [True, False]}},
        {'type': 'Choice', 'kwargs': {'name': 'criterion', 'range': ['cross_entropy']}},
        {'type': 'Continuous', 'kwargs': {'name': 'lr', 'range': [0.0001, 0.5], 'scale': 'log'}},
        {'type': 'Choice', 'kwargs': {'name': 'optimizer_momentum', 'range': [0.9]}},
        {'type': 'Choice', 'kwargs': {'name': 'optimizer_weight_decay', 'range': [0.0005]}}
        # {'type': 'Choice', 'kwargs': {'name': 'model_weights', 'range': [None]}}
        ]

config.algorithm_name = 'RandomSearch'
config.algorithm_params = {'max_num_trials': 20}

print("COMPUTATION START")
hoptim_loop(config)
print('COMPUTATION END')

