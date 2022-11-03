import torch
import torch.nn.functional as F
from src.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def create_encoder(model_config):
    model_name = model_config['encoder']['name']
    latent_dim = model_config['encoder']['latent_dim']

    model_dict = {
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
    }

    if model_name not in model_dict:
        raise ValueError(f'Invalid model name: {model_name}')

    encoder: torch.nn.Module = model_dict[model_name](num_classes=latent_dim)
    return encoder


def creaete_projector(model_config):
    input_dim = model_config['encoder']['latent_dim']
    layer_dims = [input_dim] + model_config['projector']['dims']

    layers = []
    for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(in_dim, out_dim))

    projector = torch.nn.Sequential(*layers)
    return projector


def nt_xent(
    u: torch.Tensor,  # [N, C]
    v: torch.Tensor,  # [N, C]
    temperature: float = 0.5,
):
    """
    N: batch size
    C: feature dimension
    """
    N, C = u.shape

    z = torch.cat([u, v], dim=0)  # [2N, C]
    z = F.normalize(z, p=2, dim=1)  # [2N, C]
    s = torch.matmul(z, z.t()) / temperature  # [2N, 2N] similarity matrix
    mask = torch.eye(2 * N).bool().to(z.device)  # [2N, 2N] identity matrix
    s = torch.masked_fill(
        s, mask, -float('inf'))  # fill the diagonal with negative infinity
    label = torch.cat([  # [2N]
        torch.arange(N, 2 * N),  # {N, ..., 2N - 1}
        torch.arange(N),  # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)  # NT-Xent loss
    return loss


def knn_classify(emb, cls, batch_size, n_neighbors=None):
    """Apply KNN for different K and return the maximum acc"""
    n_neighbors = n_neighbors or [1, 10, 50, 100]
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(batch_x.unsqueeze(1) - emb.unsqueeze(0),
                          dim=2,
                          p='fro')
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for k in n_neighbors:
            knn = dist.topk(k, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)
