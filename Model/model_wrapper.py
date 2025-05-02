from Pipeline.dataset import BasePytorchModelDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, train_X, train_y, test_X, test_y):
        self.model.fit(train_X, train_y)
        return {}

    def predict(self, X):
        predict = self.model.predict(X)
        return predict

    def reset(self,):
        print('Reset The model')
        if isinstance(self.model, MultiOutputRegressor):
            self.model = self.model.__class__(SVR(kernel="rbf"))
        else:
            self.model = self.model.__class__()

class PytorchModelWrapper:
    def __init__(self, model, train_config):
        self.model = model
        self.train_config = train_config
        self.logging = train_config['log_experiments']
        if self.logging :
            wandb.init(project="circuit_training", config=train_config)
   
    def reset(self,):
        print('Reset The model')
        for layers in self.model.children():
            if isinstance(layers, nn.Sequential):
                for layer in layers:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            else:
                if hasattr(layers, 'reset_parameters'):
                    layers.reset_parameters()


    def fit(self, train_X, train_y, test_X, test_y):
        train_dataset = BasePytorchModelDataset(train_X, train_y)
        test_dataset = BasePytorchModelDataset(test_X, test_y)
        train_dataloader = DataLoader(train_dataset, batch_size=100)
        test_dataloader = DataLoader(test_dataset, batch_size=100)
        train_result = self.model_train(train_dataloader, test_dataloader)
        return train_result

    def predict(self, X):
        self.model.eval()
        return self.model(torch.Tensor(X).to(self.train_config["device"])).to('cpu').detach().numpy()

    def model_train(self, train_dataloader, test_dataloader):
        import numpy as np
        import pandas as pd
        import os
        import seaborn as sns
        import matplotlib.pyplot as plt

        train_loss = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters())

        losses = []
        val_losses = []
        device = self.train_config["device"]
        self.model.to(device)

        for epoch in range(self.train_config["epochs"]):
            self.model.train()
            avg_loss = 0
            val_avg_loss = 0

            for t, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                x_var = x.float().to(device)
                y_var = y.float().to(device)

                scores = self.model(x_var)
                loss = train_loss(scores, y_var)
                loss = torch.clamp(loss, max=500000, min=-500000)
                avg_loss += (loss.item() - avg_loss) / (t + 1)
                loss.backward()
                optimizer.step()


            with torch.no_grad():
                self.model.eval()
                for t, (x, y) in enumerate(test_dataloader):
                    x_var = x.float().to(device)
                    y_var = y.float().to(device)
                    scores = self.model(x_var)
                    loss = train_loss(scores, y_var)
                    loss = torch.clamp(loss, max=500000, min=-500000)
                    val_avg_loss += (loss.item() - val_avg_loss) / (t + 1)

            losses.append(avg_loss)
            val_losses.append(val_avg_loss)

            if self.train_config["loss_per_epoch"]:
                print(f'epoch: {epoch:<4} train loss: {avg_loss:.4f}, validation loss: {val_avg_loss:.4f}')
            else:
                print(f'epoch: {epoch:<4}')

            if self.logging:
                wandb.log({'train_loss': avg_loss, 'val_loss': val_avg_loss, 'epoch': epoch})

        all_preds, all_actuals = [], []
        for x, y in test_dataloader:
            x = x.to(device).float()
            y = y.to(device).float()
            pred = self.model(x)
            all_preds.append(pred.detach().cpu().numpy())
            all_actuals.append(y.detach().cpu().numpy())




        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_actuals)

        param_names = ["C1", "WP1", "WP2", "WN1", "WN2", "WN3"]




        plt.figure(figsize=(4 * len(param_names), 4)




       for i, param in enumerate(param_names):
    y_t = y_true[:, i]
    y_p = y_pred[:, i]

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = 100 * abs((y_p - y_t) / y_t)
        rel_err = np.where(np.isfinite(rel_err), rel_err, 0)

    plt.subplot(1, len(param_names), i+1)
    sns.kdeplot(rel_err, fill=True, linewidth=2)
    plt.title(f"{param} Relative Error")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.xlim(left=0)

plt.tight_layout()
plt.savefig("graph_result/TSVA-Transformer-design-KDE.png")
plt.show()
