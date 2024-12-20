import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .estimator import DensityRatioEstimation
from .utils import evaluation_metrics


class Client:

    def __init__(self, 
                 client_name,
                 local_dataset, 
                 local_model,
                 optimizer,
                 scheduler,
                 embedder: Optional[nn.Module] = None,
                 logpath: Optional[str] = None):
        """
        """
        self.client_name    = client_name
        self.local_model    = local_model
        self.local_dataset  = local_dataset
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.embedder       = embedder
        self.logpath        = logpath
        self.estimator      = None
        self.tau_client     = None

        if self.logpath is not None:
            with open(self.logpath, 'w+') as f:
                f.write("round,mode,accuracy,precision,recall,f1\n")
        
    @torch.no_grad()
    def build_estimator(self,
                        training_frac: float,
                        gaussian_kernel_width: int,
                        lamda: float,
                        unif_min: float,
                        unif_max: float,
                        unif_dim: float,
                        unif_n: int,
                        max_num_valid: Optional[int] = None):
        """
        Build the KuLSIF estimator for the client.
        """
        # sample from the client distribution
        n_training_samples = int(len(self.local_dataset) * training_frac)
        train_samples = self.local_dataset[:n_training_samples][0] # just images
        valid_samples = self.local_dataset[n_training_samples:][0] # just images

        if max_num_valid is not None:
            # prevents exploding memory usage
            valid_samples = valid_samples[:max_num_valid]
        
        # if client has embedder, embed the images into the latent
        # space and overwrite the Uniform Distribution params with
        # params from the latent representations
        if self.embedder is not None:
            train_samples = self.embedder(train_samples)
            valid_samples = self.embedder(valid_samples)
            
            # overwrite uniform distr. params
            unif_dim = train_samples.shape[-1]
            unif_min = train_samples.min()
            unif_max = train_samples.max()
        else:
            # otherwise use the entire image flattened.
            train_samples = train_samples.reshape(train_samples.shape[0], -1)
            valid_samples = valid_samples.reshape(valid_samples.shape[0], -1)

        # sample from the uniform distribution
        uniform_samples = np.random.uniform(
            low=unif_min, 
            high=unif_max, 
            size=(unif_n, unif_dim)
        )

        # initialise the estimator
        self.estimator = DensityRatioEstimation(
            known_samples=train_samples,
            auxiliary_samples=uniform_samples,
            gaussian_kernel_width=gaussian_kernel_width,
            lamda=lamda
        )

        outcomes = self.estimator.ratio_estimator(valid_samples)
        
        # As in Section "Ablation study on thresholds of selectors",
        # the threshold is defined as the x quartile over the validation set
        # of the client. If the ratio falls below this threshold, the sample
        # is considered OOD. From Fig. 6 authors show that the first quartile
        # provides good results.
        self.tau_client = np.quantile(outcomes, q = 0.25)


    def local_training(self, 
                       loss_fn, 
                       iterations: int, 
                       batch_size: int, 
                       device: str,
                       verbose: bool = True):
        """
        Perform 1 training epoch.
        """
        self.local_model.to(device)
        self.local_model.train()
        iterations_object = range(iterations)

        if verbose:
            print("Running one local training epoch on", self.client_name)
            iterations_object = tqdm(range(iterations))
        
        for _ in iterations_object:
            images, labels = self._sample_batch(batch_size)
            images = images.to(device)
            labels = labels.to(device)
            labels_pred = self.local_model(images)[0]
            
            self.optimizer.zero_grad()
            loss = loss_fn(labels_pred, labels)
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()


    def distillation(self, 
                     loss_fn,       
                     proxy_images, 
                     proxy_softlabels: str,
                     device: str,
                     iterations: int):
        """
        Perform distillation using soft labels.
        """
        self.local_model.to(device)
        self.local_model.train()
        proxy_images = proxy_images.to(device)
        proxy_softlabels = proxy_softlabels.to(device)

        for _ in range(iterations):
            labels_pred = self.local_model(proxy_images)[2]
            self.optimizer.zero_grad()
            loss = loss_fn(labels_pred, proxy_softlabels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()


    def _sample_batch(self, batch_size: int):
        """
        Sample a mini-batch.
        """
        random_indices = np.random.choice(len(self.local_dataset), size=batch_size, replace=False)
        return self.local_dataset[random_indices]
    
    
    @torch.no_grad()
    def evaluate(self, test_loader, device: str):
        """
        Evaluate the local model.
        """
        self.local_model.eval()
        self.local_model.to(device)
        y_true = []
        y_pred = []
        for images, true_labels in test_loader:
            images = images.to(device)
            pred_labels = self.local_model(images)[1].argmax(dim=1).cpu().numpy()
            true_labels = true_labels.numpy()
            y_pred += list(pred_labels)
            y_true += list(true_labels)
        metrics = evaluation_metrics(y_pred, y_true)
        return metrics
    

    def log(self, round: int, metrics: list, mode: str):
        """
        Log the evaluation metrics.
        """
        if self.logpath is None: return
        metrics = [ str(m) for m in metrics ]
        with open(self.logpath, 'a') as f:
            line = ','.join(([str(round)] + [mode] + metrics)) + "\n" 
            f.write(line)


    def save(self, savedir: str, round: int):
        """
        Save model, optimiser and scheduler.
        """
        prefix = f"{self.client_name}_round{round}_"
        torch.save(self.local_model.state_dict(), os.path.join(savedir, f'{prefix}_mod.pt'))
        torch.save(self.optimizer.state_dict(), os.path.join(savedir,   f'{prefix}_opt.pt'))
        torch.save(self.scheduler.state_dict(), os.path.join(savedir,   f'{prefix}_lrs.pt'))