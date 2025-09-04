import os
import argparse
import warnings
from typing import List
warnings.filterwarnings("ignore")

import torch
from rich.console import Console
from rich.progress import Progress

from wfd.network import get_network, get_embedder
from wfd.client import Client
from wfd.server import (
    WeightedFDServer,
    SelectiveFDServer,
    FedMDServer
)
from wfd.utils import log_results, set_seed
from wfd.data import (
    LocalDataset, 
    LocalDatasetType, 
    ProxyDataset, 
    StoredDataset,
    SelectedProxyDataset
)

DATASET_TYPES = [
    LocalDatasetType.IID, 
    LocalDatasetType.WEAK_NON_IID, 
    LocalDatasetType.STRONG_NON_IID
]

ALGORITHMS = [
    'fed-md',
    'selective-fd',
    'weighted-fd'
]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # files and folders params
    parser.add_argument("--dataset",       type=str,    required=True, choices=['mnist', 'fashion-mnist', 'cifar10'])
    parser.add_argument("--datasets_dir",  type=str,    required=True, help="Folder where all datasets are stored")
    parser.add_argument("--dataset_type",  type=str,    required=True, choices=DATASET_TYPES)
    parser.add_argument("--output_path",   type=str,    required=True, help="Where to store the output files")
    
    # Training params
    parser.add_argument("--algorithm",          type=str,    required=True)
    parser.add_argument("--seed",               type=int,    required=True)
    parser.add_argument("--n_clients",          type=int,    default=10)
    parser.add_argument("--rounds",             type=int,    default=100)
    parser.add_argument("--start_iters",        type=int,    default=200)
    parser.add_argument("--local_iters",        type=int,    default=1)
    parser.add_argument("--distl_iters",        type=int,    default=10)
    parser.add_argument("--local_batchsize",    type=int,    default=64)
    parser.add_argument("--proxy_batchsize",    type=int,    default=512)
    parser.add_argument("--lr",                 type=float,  default=0.1)
    parser.add_argument("--proxy_fraction",     type=float,  default=1.0)
    
    # KuLSIF params
    parser.add_argument("--lamda",     type=float,  default=0.0632)
    parser.add_argument("--gauss",     type=int,    default=5)
    parser.add_argument("--tfrac",     type=float,  default=0.05)
    parser.add_argument("--nunif",     type=int,    default=250)
    parser.add_argument("--mnval", type=int, default=None, help='avoid memory explosion')
    args = parser.parse_args()
    
    assert args.algorithm in ALGORITHMS, "algorithm not found"
    
    data_path = os.path.join(args.datasets_dir, args.dataset + ".pt")
    assert os.path.exists(data_path), "Dataset not found in path " + data_path

    assert os.path.exists(args.output_path), "Output folder does not exist"
    logs_path = os.path.join(args.output_path, "logs")    
    save_path = os.path.join(args.output_path, "save")
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Setting seed, device and console
    torch.manual_seed(0)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    console = Console()
    set_seed(args.seed)
    
    #=====================================================================
    # Initialise clients
    #=====================================================================

    embedder = None if args.algorithm != 'selective-fd' else get_embedder(args.dataset)
    if embedder is not None: embedder.eval()

    with console.status("[bold green]Initialising the clients.", spinner='aesthetic') as status:
        clients: List[Client] = []
        for client_idx in range(args.n_clients):
            model = get_network(args.dataset)
            optim = torch.optim.SGD(model.parameters(), lr=args.lr)
            sched = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.7)
            client_name = f'client-{client_idx}'
            clients.append(Client(client_name=client_name,
                                  logpath=os.path.join(logs_path, client_name + ".csv"),
                                  local_dataset=LocalDataset(data_path, client_idx, dataset_type=args.dataset_type), 
                                  local_model=model, 
                                  optimizer=optim, 
                                  scheduler=sched,
                                  embedder=embedder))

    #=====================================================================
    # Initialise the proxy dataset
    #=====================================================================

    console.print("[bold green]Initialising the proxy dataset")
    proxy_dataset = ProxyDataset(data_path, fraction=args.proxy_fraction)
    
    #=====================================================================
    # Instantiate server
    #=====================================================================

    console.print("[bold green]Creating the aggregator server")

    if args.algorithm == 'selective-fd':

        with console.status(
            "[bold green]Applying selective-fd to the proxy dataset (this might take a while...)",
            spinner='aesthetic') as status:
            
            selected_proxy_dataset = SelectedProxyDataset(clients=clients,
                                                          proxy_dataset=proxy_dataset, 
                                                          dre_embedder=embedder,
                                                          dre_training_frac=args.tfrac, 
                                                          dre_gaussian_kernel_width=args.gauss,
                                                          dre_lamda=args.lamda,
                                                          dre_unif_n=args.nunif,
                                                          dre_max_num_valid=args.mnval,
                                                          cache=True,
                                                          cache_path=os.path.join(save_path, f'{args.dataset}-spd.pt'))
    
            server = SelectiveFDServer(clients, selected_proxy_dataset)

    elif args.algorithm == 'weighted-fd':
        server = WeightedFDServer(clients, proxy_dataset)

    elif args.algorithm == 'fed-md':        
        server = FedMDServer(clients, proxy_dataset)
    else:
        raise Exception("Algorithm not supported.")

    #=====================================================================
    # Instantiate validation and test sets
    #=====================================================================

    validset = StoredDataset(data_path, key='valid_set')
    valid_loader = torch.utils.data.DataLoader(validset, shuffle=False, batch_size=256)

    testset = StoredDataset(data_path, key='test_set')
    test_loader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=256)


    #=====================================================================
    # Start the training!
    #=====================================================================    

    console.print("[bold blue]200 training steps on the local datasets.")

    for client in clients:
        client.local_training(
            loss_fn=torch.nn.functional.nll_loss,
            iterations=args.start_iters,
            batch_size=args.local_batchsize,
            device=device,
            verbose=False
        )

    with Progress() as progress:

        task = progress.add_task("Training.", total=args.rounds)

        for round in range(args.rounds):
            
            for client in clients:
                client.local_training(
                    loss_fn=torch.nn.functional.nll_loss,
                    iterations=args.local_iters,
                    batch_size=args.local_batchsize,
                    device=device,
                    verbose=False
                )

            proxy_images, proxy_softlabels, _ = \
                server.generate_soft_labels(batch_size=args.proxy_batchsize, device=device)

            for client in clients:
                client.distillation(
                    loss_fn=torch.nn.functional.cross_entropy,
                    iterations=args.distl_iters,
                    proxy_images=proxy_images,
                    proxy_softlabels=proxy_softlabels,
                    device=device,
                )

            progress.console.print(f"[bold orange] evaluating the clients")
            _     = log_results(round, clients, valid_loader, 'valid', device)
            table = log_results(round, clients, test_loader, 'test', device)
            console.print(table)                    
            progress.advance(task)
    
    