import os
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .client import Client
from .utils import evaluation_metrics


class LocalDatasetType:
    IID = 'local_sets_iid'
    WEAK_NON_IID = 'local_sets_wni'
    STRONG_NON_IID = 'local_sets_hni'


class LocalDataset:

    def __init__(self, dataset_path: str, client_idx: int, dataset_type: LocalDatasetType):
        """
        """
        dictionary = torch.load(dataset_path, weights_only=True)[dataset_type]
        self.local_images = dictionary[client_idx]['images']
        self.local_labels = dictionary[client_idx]['labels']

    def __len__(self):
        return self.local_images.shape[0]

    def __getitem__(self, idx):
        image = self.local_images[idx]
        label = self.local_labels[idx]
        return image, label


class StoredDataset:

    def __init__(self, dataset_path: str, key: str):
        """
        """
        dictionary = torch.load(dataset_path, weights_only=True)[key]
        self.images = dictionary['images']
        self.labels = dictionary['labels']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


class ProxyDataset:

    def __init__(self, dataset_path: str, fraction: float = 1.):
        """
        The dictionary contains a key for each class, and the values are
        the images that belong to that class. All the images from all the 
        classes are aggregated into the proxy dataset.
        """
        dictionary = torch.load(dataset_path, weights_only=True)['proxy_set']
        self.proxy_images = dictionary['images']
        self.proxy_labels = dictionary['labels']
        if fraction < 1.: self._reduce_size(fraction)

    def _reduce_size(self, fraction):
        new_size = int(len(self.proxy_images) * fraction)
        self.proxy_images = self.proxy_images[:new_size]
        self.proxy_labels = self.proxy_labels[:new_size]

    def __len__(self):
        return self.proxy_images.shape[0]

    def __getitem__(self, idx):
        image = self.proxy_images[idx]
        label = self.proxy_labels[idx]
        return image, label
    
    @torch.no_grad()
    def to_label_clusters(self):
        """
        This is a utilify function that converts this dataset into
        a dictionary where the keys are the classes, and the values
        are the images flattened in one dimension. This will be used
        in the preparation of the `SelectedProxyDataset`.
        """
        clusters = {}
        unique_labels = torch.unique(self.proxy_labels)
        for label in unique_labels:
            clusters[label.item()] = self.proxy_images[ self.proxy_labels == label ]
        return clusters

    def get_image_shape(self):
        return self.proxy_images[0].shape
    
    def get_flattened_image_shape(self):
        return np.prod(self.proxy_images[0].shape)


class SelectedProxyDataset(Dataset):
    """
    """

    def __init__(self,
                 clients: List[Client],
                 proxy_dataset: ProxyDataset,
                 dre_embedder: Optional[nn.Module] = None,
                 dre_training_frac: float = 0.05,
                 dre_gaussian_kernel_width: int = 5,
                 dre_lamda: float = 0.06324555320336758,
                 dre_unif_n: int = 250,
                 dre_max_num_valid: Optional[int] = None,
                 cache: bool = True,
                 cache_path: Optional[str] = None,):
        """
        """
        self.clients = clients
        self.proxy_dataset = proxy_dataset
        
        # parameters for the KuLSIF model
        self.dre_embedder               = dre_embedder
        self.dre_training_frac          = dre_training_frac
        self.dre_gaussian_kernel_width  = dre_gaussian_kernel_width
        self.dre_lamda                  = dre_lamda
        self.dre_unif_n                 = dre_unif_n
        self.dre_max_num_valid          = dre_max_num_valid

        if cache_path is not None and os.path.exists(cache_path):
            data = torch.load(cache_path)
            self.labels_x_images = data["labels_x_images"] 
            self.labels_x_clients_votes = data["labels_x_clients_votes"]
        else:
            self._construct()

        if cache and not os.path.exists(cache_path):
            torch.save({
                'labels_x_images': self.labels_x_images,
                'labels_x_clients_votes': self.labels_x_clients_votes
            }, cache_path)

        # the last step to create the dataset...
        images = []
        labels_ensemble = []
        for label, label_images in self.labels_x_images.items():
            images.append(label_images)
            votes = self.labels_x_clients_votes[label]
            labels_ensemble.append(torch.from_numpy(votes))
        
        # this will be a N x C x H x W matrix (N=#samples, C=#channels, H=height, W=width)
        self.images = torch.concat(images, dim=0)
        # this will be a N x K matrix (N=#samples, K=#clients)
        self.voting = torch.concat(labels_ensemble, dim=0)
        # this will be a N x 1 matrix (N=#samples)
        self.labels = torch.concat([torch.full((len(v),), k) for k, v in self.labels_x_images.items()], dim=0)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images = self.images[idx]
        voting  = self.voting[idx]
        labels = self.labels[idx]
        return images, voting, labels

    def _construct(self):
        """
        """
        # building the client side selector for each client.
        # (read paper at page 8)
        for ith_client in self.clients:

            ith_client.build_estimator(
                training_frac=self.dre_training_frac,
                gaussian_kernel_width=self.dre_gaussian_kernel_width,
                lamda=self.dre_lamda,
                unif_min=self.proxy_dataset.proxy_images.min(),
                unif_max=self.proxy_dataset.proxy_images.max(),
                unif_dim=self.proxy_dataset.get_flattened_image_shape(),
                unif_n=self.dre_unif_n,
                max_num_valid=self.dre_max_num_valid
            )

        # creating a dictionary where the keys are the classes
        # and the values are flattened images (1d) that belong 
        # to that class.
        proxy_clusters = self.proxy_dataset.to_label_clusters()

        # This dictionary has one key for each label, and 
        # the corresponding value is a tensor of non-ood images
        # meaning that the image is in the distribution of at least
        # on client
        labels_x_images = {}

        # following the explanation above, this dictionary
        # indicates for which client the image is in distribution. 
        labels_x_clients_votes = {}

        for label, label_data in proxy_clusters.items():
            
            # if the estimator uses an embedder, then project the images into the 
            # latent space with the same embedder, otherwise just flatten.
            if self.dre_embedder is None:
                label_data_flattened = label_data.reshape(label_data.shape[0], -1).numpy()
            else:
                label_data_flattened = self.dre_embedder(label_data).numpy()

            # needed to aggregate the results
            clients_binary_outcomes = {}

            for client in self.clients:
                
                # estimate the density ratio on the data from the current label
                # for this client. Then classify as "In Distribution" if the outcome
                # is higher than the first quartile (saved in the evaluator class)
                outcomes = client.estimator.ratio_estimator(label_data_flattened)
                outcomes_binary = outcomes > client.tau_client
                clients_binary_outcomes[client.client_name] = outcomes_binary

            # This creates a N x K matrix where N = label_data.shape[0] and K is
            # the number of clients.
            aggregation = np.array([ b for b in clients_binary_outcomes.values() ])
            aggregation = np.swapaxes(aggregation, axis1=0, axis2=1)

            # If we sum over the client axis we can analyse if
            # an image is OOD for all the clients. In this case, 
            # we discard the image. All the images in the `selected_list`
            # ad "in distribution" for some client.
            aggregation_sum = np.sum(aggregation, axis = 1)
            selected_list = np.nonzero(aggregation_sum)

            # store non-ood images and the respective clients votes
            labels_x_images[label] = label_data[selected_list]
            labels_x_clients_votes[label] =  aggregation[selected_list] / np.reshape(aggregation_sum[selected_list], (-1,1))

        # store this into the class.
        self.labels_x_images = labels_x_images 
        self.labels_x_clients_votes = labels_x_clients_votes