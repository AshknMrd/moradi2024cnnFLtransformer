"""SwinUNetr FL: A Flower / PyTorch app."""

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

from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete


class FlowerClient(NumPyClient):
    """Federated learning client for SwinUNETR using Flower."""

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

    net = SwinUNETR(
        in_channels=3,
        out_channels=2,
        img_size=(256, 256, 32),
        feature_size=48,
        drop_rate=0.02,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        use_checkpoint=True,
        use_v2=True,
    )

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)