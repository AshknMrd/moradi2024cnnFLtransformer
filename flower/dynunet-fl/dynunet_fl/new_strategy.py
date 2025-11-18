# Creating customized strategy
import os
from datetime import date

import torch
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from monai.networks.nets import DynUNet

from first_example.task import set_weights


# Output directory
today = date.today().isoformat()
save_dir = f"./flwr_output_dynunet/{today}"
os.makedirs(save_dir, exist_ok=True)


class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_to_save = {}
        self.train_loss_log = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:

        # Perform FedAvg aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Convert Flower parameters → numpy arrays
        ndarrays = parameters_to_ndarrays(aggregated_params)

        # Recreate and load global model
        model = DynUNet(
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
        set_weights(model, ndarrays)

        # Save global model snapshot
        model_path = os.path.join(save_dir, f"global_model_round{server_round}.pt")
        torch.save(model.state_dict(), model_path)

        return aggregated_params, aggregated_metrics