# -*- coding: utf-8 -*-
import datetime
# import sys

import sherpa

from serialize import load_json
from deep_learning import prepare_config, run_training, DEFAULT_CONFIG
from data import AUG_LIST_FLIPS

#%%
class HOptimConfig:
    train_config = DEFAULT_CONFIG
    train_config['training']['num_epochs'] = 3
    train_config['training']['batch_size'] = 12

    paths_file = 'paths.json'

    params = [
        {'type': 'Choice', 'kwargs': {'name': 'model_name', 'range': ['resnet18', 'resnet34']}}, #, 'resnet50', 'densenet121', 'efficientnet_b0']}},
        {'type': 'Choice', 'kwargs': {'name': 'model_pretrained', 'range': [True]}},
        {'type': 'Choice', 'kwargs': {'name': 'augmentation_train', 'range': [None, AUG_LIST_FLIPS]}},
        {'type': 'Choice', 'kwargs': {'name': 'criterion', 'range': ['cross_entropy']}},
        {'type': 'Continuous', 'kwargs': {'name': 'lr', 'range': [0.0001, 0.5], 'scale': 'log'}},
        {'type': 'Choice', 'kwargs': {'name': 'optimizer_momentum', 'range': [0.9]}},
        {'type': 'Choice', 'kwargs': {'name': 'optimizer_weight_decay', 'range': [0.0005]}}
        # {'type': 'Choice', 'kwargs': {'name': 'model_weights', 'range': [None]}}
        ]

    algorithm_name = 'RandomSearch'
    algorithm_params = {'max_num_trials': 20}

    log_dir = load_json(paths_file)['output']



#%%
def config_from_params(base_config, paths_file,
                       model_name, model_pretrained,
                       augmentation_train,
                       criterion, lr,
                       optimizer_momentum, optimizer_weight_decay):

    config = prepare_config(base_config, paths_file)

    config['model']['name'] = model_name
    config['model']['pretrained'] = model_pretrained
    config['data']['augmentations']['train'] = augmentation_train
    config['training']['criterion'] = criterion
    config['training']['optimizer']['kwargs']['lr'] = lr

    config['training']['optimizer']['kwargs']['momentum'] = optimizer_momentum
    config['training']['optimizer']['kwargs']['weight_decay'] = optimizer_weight_decay

    return config

#%%
def hoptim_loop(hyperparam_config):
    parameters = [sherpa.__getattribute__(p['type'])(**p['kwargs']) for p in hyperparam_config.params]

    algorithm = sherpa.algorithms.__getattribute__(hyperparam_config.algorithm_name)(**hyperparam_config.algorithm_params)

    study = sherpa.Study(parameters=parameters,
                 algorithm=algorithm,
                 lower_is_better=False,
                 disable_dashboard= True, #sys.platform == 'win32',
                 output_dir=hyperparam_config.log_dir
                 )

    for trial in study:
        config = config_from_params(hyperparam_config.train_config,
                                    hyperparam_config.paths_file,
                                    **trial.parameters)
        start_time = datetime.datetime.now()
        metrics = run_training(config)

        study.add_observation(trial=trial,
                          objective=metrics['val_accuracy'].cpu().item(),
                          context = {
                              'run_label': config['run_label'],
                              'val_accuracy': metrics['val_accuracy'].cpu().item(),
                              'val_loss': metrics['val_loss'].cpu().item(),
                              'test_loss': metrics['test_loss'].cpu().item(),
                              'test_accuracy': metrics['test_accuracy'].cpu().item(),
                              'train_loss': metrics['train_loss'].cpu().item(),
                              'train_accuracy': metrics['train_accuracy'].cpu().item(),
                              'time_s': (datetime.datetime.now() - start_time).total_seconds()
                              }
                          )
        study.finalize(trial)

        study.save()
#%%
if __name__ == '__main__':
    hoptim_loop(HOptimConfig)