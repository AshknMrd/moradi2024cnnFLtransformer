"""DynUNet FL: A Flower / PyTorch app."""

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from first_example.task import (
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)

from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete


class FlowerClient(NumPyClient):
    """Federated learning client for DynUNet using Flower."""

    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.post_label = AsDiscrete(to_onehot=2)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)

        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)

        loss, dice = test(
            self.net,
            self.valloader,
            self.device,
            self.post_label,
            self.post_pred,
        )

        return loss, len(self.valloader.dataset), {"dice": dice}


def client_fn(context: Context):
    """Create and configure a Flower client."""
    
    net = DynUNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=2,
        filters=[32, 64, 128, 256, 320],
        kernel_size=[3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2],
        norm_name="INSTANCE",
        res_block=True,
    )
    

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)