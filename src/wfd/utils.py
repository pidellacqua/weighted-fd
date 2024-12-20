import torch
import random
import numpy as np

from rich.table import Table
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


def evaluation_metrics(y_pred, y_true):
    """
    classification eval metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # 'weighted' handles multi-class
    recall = recall_score(y_true, y_pred, average='weighted')        # 'weighted' handles multi-class
    f1 = f1_score(y_true, y_pred, average='weighted')                # 'weighted' handles multi-class
    return accuracy, precision, recall, f1


def log_results(_round, clients, dataloader, mode, device):
    results_data = []
    for client in clients:
        metrics = client.evaluate(dataloader, device)
        client.log(_round, metrics, mode)
        accuracy, precision, recall, f1 = metrics
        results_data.append([client.client_name, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    table = Table(title=f"Client Evaluation Metrics for _round {_round}")
    table.add_column("Client", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="magenta")
    table.add_column("F1 Score", style="magenta")
    for row in results_data: table.add_row(*row)
    return table


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() == True:
        torch.cuda.manual_seed(seed)