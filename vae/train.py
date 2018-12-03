import torch.nn.functional as F

def update(data, model, optimizer):
    optimizer.zero_grad()
    sample, mu, logvar, recon = model(data)
    reconstruction = F.binary_cross_entropy(recon, data, reduction='sum')
    variational = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    optimizer.step()
    loss = (reconstruction, variational)
    return loss

