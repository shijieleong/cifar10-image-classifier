import torch

def save_checkpoint(net):
    torch.save(net.state_dict(), './models/trained_model.pth')
    print("Checkpoint saved!")


def load_checkpoint(net, checkpoint_path):
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return net
