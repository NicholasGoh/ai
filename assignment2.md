---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: ai
    language: python
    name: ai
---

```python
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import tqdm
torch.manual_seed(0);
```

- 48 numerical features
- 1 target

```python
data = scipy.io.loadmat('OQC.mat')
data = pd.DataFrame(data.get('data'))
data.head()
```

- small dataset
    - model cannot have too many parameters
    - otherwise very likely to overfit

```python
features = data.iloc[:, :-1].values
targets = data.iloc[:, -1].values
features.shape, targets.shape
```

```python
features = torch.tensor(features).float() # float64 to float32
targets = torch.tensor(targets).long() # cast to int64
features.dtype, targets.dtype
```

- 70/30 `train`/`valid` split

```python
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(features, targets, test_size= 0.30, random_state=42)
```

Helper function to:
- retrieve batches of (x, y) and pass them to gpu with `collate_fn`

```python
from torch.utils.data.dataloader import default_collate

def init_loaders(batch_size=1, device='cuda'):
    # collate_fn moves each batch to gpu
    train_loader = DataLoader(list(zip(x_train, y_train)),
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda x: list(x_.to(device) for x_ in default_collate(x)))
    valid_loader = DataLoader(list(zip(x_valid, y_valid)),
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=lambda x: list(x_.to(device) for x_ in default_collate(x)))
    return dict(train=train_loader, valid=valid_loader)
```

```python
dataloaders = init_loaders()
```

Peek at data:
- 48 features
- 1 target
- all on gpu (`cuda`)

```python
for x, y in dataloaders['valid']:
    break
x, y
```

# Task 1
Helper function for 1 Epoch
- for `train` and `valid`
    - for each minibatch
        - pass minibatch through model
        - get predictions
        - calculate loss, acc
        - if `train` then update weights
        - log results

```python
def train_valid(model,
                dataloaders,
                optimizer,
                epoch,
                history,
                log_interval=10 ** 3,
                verbose=False,
                device='cuda'):
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train() # enable stuff like dropout
        else:
            model.eval() # disable stuff like dropout
            
        running_correct = 0
        running_loss = 0
        running_acc = 0
        dataloader = dataloaders[phase]
        progress_bar = tqdm.notebook.tqdm(dataloader) if verbose else dataloader
        
        for batch_idx, (feature, target) in enumerate(progress_bar):
            
            if batch_idx == 0: # last
                batch_size = len(target) # last len(target) may be < batch_size
                
            optimizer.zero_grad() # torch accumulates grads (easier for nets like RNNs)
            
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                output = model(feature)
                _, preds = torch.max(output, dim=-1)
                loss = F.cross_entropy(output, target, reduction='sum') # use once only so use lower level function
            
                if phase == 'train':
                    loss.backward() # computes (partial) derivatives
                    optimizer.step() # w -= alpha * (partial) derivatives

            running_loss += loss.item()
            correct = torch.sum(preds == target).cpu().item() # pass to cpu for use later
            running_correct += correct
            running_acc += correct / len(target)

            if verbose and batch_idx % log_interval == 0:
                log = '------\tEpoch: {} | {}_loss: {:.6f} | {}_acc : {:.6f}\t------'.format(
                    epoch, phase, loss.item(), phase, correct / len(target))
                progress_bar.set_description(log)
                progress_bar.refresh()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_correct / (len(dataloader) * batch_size)
        history.get(f'{phase}_loss').append(epoch_loss)
        history.get(f'{phase}_acc').append(epoch_acc)
    
    return history
```

- 48 input features
- 3 class classification problem

```python
input_size = len(features[0])
num_classes = len(np.unique(targets))
input_size, num_classes
```

Helper class to construct model with varying neurons (width) and hidden layers (depth)

```python
class Model(nn.Module):
    def __init__(self, neurons, hidden_layers):
        super().__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.relu = nn.ReLU(inplace=True)

        layers = list()
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.ReLU(inplace=True))
            
        self.layers = nn.Sequential(*layers) if layers != list() else list()
        self.clf = nn.Linear(neurons, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.clf(x)
        return x
```

Helper function to:
- create model
- run `train_valid` for each epoch

```python
import torchinfo

def run_network(neurons=8,
                batch_size=32,
                hidden_layers=1,
                lr=.01,
                epochs=10,
                device='cuda',
                verbose=False,
                summary=False):
    
    dataloaders = init_loaders(batch_size, device)
    model = Model(neurons=neurons, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = {k: list() for k in ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']}
    
    if summary:
        print(torchinfo.summary(model, input_size=[batch_size, input_size]))

    for epoch in range(1, epochs + 1):
        history = train_valid(model=model,
                              dataloaders=dataloaders,
                              optimizer=optimizer,
                              epoch=epoch,
                              history=history,
                              verbose=verbose,
                              device=device)
    return history
```

```python
%%time

vanilla_results = run_network(summary=True, verbose=True)
```

Helper function to visualize loss and accuracy

```python
import matplotlib.pyplot as plt

def plot(history, suptitle_label=''):
    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    f.subplots_adjust(top=.75 if suptitle_label == '' else .7)

    train_acc = history['train_acc']
    train_loss = history['train_loss']
    valid_acc = history['valid_acc']
    valid_loss = history['valid_loss']
    
    axes[1].plot(history['train_acc'])
    axes[1].plot(history['valid_acc'])
    axes[1].set_title('Model acc')
    axes[1].set(ylabel = 'acc', xlabel = 'Epoch')
    axes[1].legend(['Train', 'valid'], loc='upper left')
    axes[0].plot(history['train_loss'])
    axes[0].plot(history['valid_loss'])
    axes[0].set_title('Model loss')
    axes[0].set(ylabel = 'loss', xlabel = 'Epoch')
    axes[0].legend(['Train', 'valid'], loc='upper left')
    axes[0].grid()
    axes[1].grid()
    
    title = (
        f'{suptitle_label}\n' +
        'Min Training loss: {:.{}f}\n'.format(np.min(train_loss), 3) +
        'Max Training acc: {:.{}f}\n'.format(np.max(train_acc), 3) +
        'Min Validation loss: {:.{}f}\n'.format(np.min(valid_loss), 3) +
        'Max Validation acc: {:.{}f}\n'.format(np.max(valid_acc), 3)
    )
    f.suptitle(title)
```

For the default hyperparameters:
- performance is decent

The following will be varied to see if this performance can be improved
- `neurons` (width), `hidden_layers` (depth)
- `lr`
- `batch_size`

```python
plot(vanilla_results, 'Vanilla Model')
```

## Task 2
You are asked to study the effect of network structure: hidden nodes, hidden layers to the classification
performance. That is, you try different network configurations and understand the patterns. Your
experiments have to be well-documented in your Jupyter notebook file and your report. It has to cover
different aspects of network configurations such as shallow network, wide network, deep network etc.

## First we vary width

```python
%%time

neurons = [2 ** x for x in list(range(2, 9))]

for neuron in neurons:
    history = run_network(neurons=neuron)
    plot(history, f'{neuron} neurons')
```

```python
%%time

neurons = 64
layers = list(range(1, 4))

for hidden_layers in layers:
    history = run_network(neurons=neurons,
                          hidden_layers=hidden_layers)
    plot(history, f'{hidden_layers} hidden_layers')
```

## Then we vary depth


## Task 3
You are asked to study the effect of learning rates. As with Task 2, your experiments have to be well documented. You need to give correct conclusion and give suggestion how learning rates should be
set. This includes possible adaptive learning rates where the value increases or decreases as the
increase of epochs. 

```python
%%time

neurons = 64
lrs = [10 ** -x for x in reversed(list(range(1, 5)))] + [.5]

for lr in lrs:
    history = run_network(neurons=neuron,
                          lr=lr)
    plot(history, f'Learning Rate: {lr}')
```

## Task 4
You are asked to study the effect of mini-batch size. You can set mini-batch size to be 1 (stochastic
gradient descent), N (batch gradient descent) or any other size. The most important aspect is to be
conclusive with your finding. The mini-batch size really depends on the problem size. 

```python
%%time

neurons = 64
batch_sizes = [2 ** x for x in list(range(6))]

for batch_size in batch_sizes:
    history = run_network(neurons=neuron,
                          lr=lr,
                          batch_size=batch_size)
    plot(history, f'Batch Size: {batch_size}')
```

```python

```
