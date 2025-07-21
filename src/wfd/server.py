from typing import List

import torch
import numpy as np

from .client import Client
from .data import ProxyDataset, SelectedProxyDataset


class SelectiveFDServer:

    def __init__(self, clients: List[Client], selected_proxy_dataset: SelectedProxyDataset, tau_server: int = 2):
        self.clients = clients
        self.selected_proxy_dataset = selected_proxy_dataset
        self.threhsold = 1 - (tau_server / 2)


    @torch.no_grad()
    def generate_soft_labels(self, batch_size: int, device: str):
        """
        Aggregate the clients predictions on the proxy dataset to create soft labels.
        """
        random_indices = np.random.choice(len(self.selected_proxy_dataset), size=batch_size, replace=False)
        images, votes, labels = self.selected_proxy_dataset[random_indices]
        images, votes, labels = images.to(device), votes.to(device), labels.to(device)


        softlabels = 0.0
        for i, client in enumerate(self.clients):
            client.local_model.to(device)
            client.local_model.eval()
            _, client_labels, _ = client.local_model(images)

            # the following is going to mask based on the client's votes:
            client_labels *= torch.reshape(votes[:,i], (-1,1)).to(device)

            softlabels += client_labels

        # server-side selection
        selection = self._filter(softlabels)
        images, softlabels, labels = images[selection], softlabels[selection], labels[selection]

        return images, softlabels, labels
    
    
    def _filter(self, softlabels):
        """
        """
        # As stated in the paper, "he confidence score refers to the maximum probability
        # of the logits in the classification task, which reflects the reliability of the
        # prediction". We must select the softlabels with CS >= 1 - (tau_server / 2), which
        # implies no filtering for self.tau_server = 2.
        confidence_scores = softlabels.max(dim=1).values
        return confidence_scores >= self.threhsold
    

class WeightedFDServer:

    def __init__(self, clients: List[Client], proxy_dataset: ProxyDataset):
        self.clients = clients
        self.proxy_dataset = proxy_dataset


    @torch.no_grad()
    def generate_soft_labels(self, batch_size: int, device: str):
        """
        Aggregate the clients predictions on the proxy dataset to create soft labels.
        """
        num_classes = 10
        num_clients = len(self.clients)
        random_indices = np.random.choice(len(self.proxy_dataset), size=batch_size, replace=False)
        images, labels = self.proxy_dataset[random_indices]
        images, labels = images.to(device), labels.to(device)

        
        kl_results = torch.zeros((images.shape[0], 1,  num_clients), device=device)
        softlabels = torch.zeros((images.shape[0], num_clients, num_classes), device=device)
        onehot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(device)


        for i, client in enumerate(self.clients):

            client.local_model.to(device)
            client.local_model.eval()

            client_labels = client.local_model(images)[1]
            client_klweig = self._kl(onehot_labels, client_labels)

            softlabels[:, i, :] = client_labels
            kl_results[:, 0, i] = client_klweig
            
        softlabels = torch.bmm(kl_results, softlabels)
        softlabels = torch.nn.functional.softmax(softlabels, dim=-1).squeeze(1)
        return images, softlabels, labels


    def _kl(self, hard, soft):
        values = (hard * (soft)).sum(dim=-1).log()
        values = torch.where(values == 0, torch.tensor(1e-10, device=soft.device), values)
        return -1 /  values
    
    
class FedMDServer:
    
    def __init__(self, clients: List[Client], proxy_dataset: ProxyDataset):
        self.clients = clients
        self.proxy_dataset = proxy_dataset


    @torch.no_grad()
    def generate_soft_labels(self, batch_size: int, device: str):
        """
        Aggregate the clients predictions on the proxy dataset to create soft labels.
        """
        num_classes = 10
        num_clients = len(self.clients)
        random_indices = np.random.choice(len(self.proxy_dataset), size=batch_size, replace=False)
        images, labels = self.proxy_dataset[random_indices]
        images, labels = images.to(device), labels.to(device)

        softlabels = torch.zeros((images.shape[0], num_classes), device=images.device)

        for client in self.clients:
            client.local_model.to(device)
            client.local_model.eval()

            client_labels = client.local_model(images)[1]
            softlabels += client_labels / num_clients
            
        return images, softlabels, labels
