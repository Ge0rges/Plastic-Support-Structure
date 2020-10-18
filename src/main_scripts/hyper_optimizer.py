"""
File contains functions to search for best parameters.
"""
import numpy as np
import ray
import torch
import os

from multiprocessing import cpu_count
from sklearn.decomposition import PCA
from src.main_scripts.pss_trainer import PSSTrainer
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.trial import ExportFormat
from ray.tune.utils import validate_save_restore


class PytorchTrainable(tune.Trainable):
    """
    Train a Pytorch net with Trainable and PopulationBasedTraining scheduler.
    Finds hypers for one task.
    """

    def _setup(self, config):
        trainer_args = config.get("DENTrainerArgs")  # Type: list

        self.trainer = PSSTrainer(*trainer_args)

        self.trainer.optimizer = torch.optim.SGD(
            self.trainer.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))

        self.trainer.expand_by_k = config.get("expand_by_k", 10)
        self.trainer.drift_threshold = config.get("drift_threshold", 0.02)

        if hasattr(self.trainer.penalty, "l1_coeff"):
            self.trainer.penalty.l1_coeff = config.get("l1_coeff", 0)

        if hasattr(self.trainer.penalty, "l2_coeff"):
            self.trainer.penalty.l2_coeff = config.get("l2_coeff", 0)

    def _train(self):
        # Do one epoch for all tasks
        for i in range(self.trainer.number_of_tasks):
            self.trainer.train_tasks([i], 5, (not i == 0))

        loss_err_list = self.trainer.test_model(tasks=list(range(self.trainer.number_of_tasks)))

        mean_err = 0
        mean_loss = 0

        for loss, err in loss_err_list:
            mean_err += err
            mean_loss += loss

        mean_loss /= len(loss_err_list)
        mean_err /= len(loss_err_list)

        return {"mean_accuracy": mean_err, "mean_loss": mean_loss}

    def _save(self, checkpoint_dir):
        return self.trainer.save_model("model.pth", checkpoint_dir)

    def _restore(self, checkpoint_path):
        self.trainer.load_model(checkpoint_path)

    def _export_model(self, export_formats, export_dir):
        if export_formats == [ExportFormat.MODEL]:
            path = os.path.join(export_dir, "exported_actionnet.pt")
            torch.save(self.trainer.model.state_dict(), path)
            return {export_formats[0]: path}

        raise ValueError("unexpected formats: " + str(export_formats))

    def reset_config(self, new_config):
        for param_group in self.trainer.optimizer.param_groups:
            if "lr" in new_config:
                param_group["lr"] = new_config["lr"]
            if "momentum" in new_config:
                param_group["momentum"] = new_config["momentum"]

        self.config = new_config
        return True


class OptimizerController:
    """
    An interface between our experiments and the trainable API of Ray tune.
    Makes the required parameters accessible to Trainable.
    Exposes our layer sizes constructor function.
    """

    def __init__(self, device, data_loaders, criterion, penalty, error_function, number_of_tasks, drift_threshold, encoder_in,
                 hidden_encoder_layers, hidden_action_layers, action_out, core_invariant_size=None):
        """
        Creates a controller object.
        """

        # Sizes
        self.encoder_in = encoder_in
        self.hidden_encoder = hidden_encoder_layers
        self.hidden_action = hidden_action_layers
        self.action_out = action_out
        self.core_invariant_size = core_invariant_size
        self.device = device

        # Get the CI size
        if core_invariant_size is None:
            self.core_invariant_size = self.pca_dataset(data_loaders, threshold=0.96)

        # Build the network arch
        sizes = self.construct_network_sizes()

        # Setup trainer. LR, Momentum, expand_by_k will be replaced.
        self.trainer_args = [data_loaders, sizes, 0, 0, criterion, penalty, 0, device, error_function,
                                  number_of_tasks, drift_threshold]

    def __call__(self, num_workers: int):
        ray.init(num_gpus=torch.cuda.device_count())

        # Default config
        config = {
                "lr": np.random.uniform(0.001, 1),
                "momentum": np.random.uniform(0, 1),
                "DENTrainerArgs": self.trainer_args,
                "expand_by_k": int(np.random.uniform(1, 20)),
                "l1_coeff": np.random.uniform(1e-20, 0),
                "l2_coeff": np.random.uniform(1e-20, 0),
                "drift_threshold": np.random.uniform(0.001, 5)
        }

        # check if PytorchTrainable will save/restore correctly before execution
        # validate_save_restore(PytorchTrainable, config=config, num_gpus=torch.cuda.device_count())
        # validate_save_restore(PytorchTrainable, config=config, use_object_store=True, num_gpus=torch.cuda.device_count())

        # PBT Params
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="mean_accuracy",
            mode="max",
            perturbation_interval=5,
            hyperparam_mutations={
                # distribution for resampling
                "lr": lambda: np.random.uniform(0.0001, 1),
                "momentum": lambda: np.random.uniform(0, 0.99),
                "expand_by_k": lambda: int(np.random.uniform(1, 20)),
                "l1_coeff": lambda: np.random.uniform(1e-20, 1),
                "l2_coeff": lambda: np.random.uniform(1e-20, 1),
                "drift_threshold": lambda: np.random.uniform(0.001, 5)
            })

        # Tune params
        class CustomStopper(tune.Stopper):
            def __init__(self):
                self.should_stop = False

            def __call__(self, trial_id, result):
                max_iter = 100
                if not self.should_stop and result["mean_accuracy"] > 0.96:
                    self.should_stop = True
                return self.should_stop or result["training_iteration"] >= max_iter

            def stop_all(self):
                return self.should_stop

        stopper = CustomStopper()

        needed_cpu = max(1, cpu_count() / num_workers)
        needed_gpu = max(0.1, torch.cuda.device_count() / num_workers)
        analysis = tune.run(
            PytorchTrainable,
            name="pbt_test",
            scheduler=scheduler,
            reuse_actors=True,
            resources_per_trial={"cpu": needed_cpu, "gpu": needed_gpu},
            verbose=2,
            stop=stopper,
            export_formats=[ExportFormat.MODEL],
            checkpoint_score_attr="mean_accuracy",
            checkpoint_freq=5,
            keep_checkpoints_num=4,
            num_samples=num_workers,
            config=config)

        # Retrieve results
        best_trial = analysis.get_best_trial("mean_accuracy")
        best_checkpoint = max(analysis.get_trial_checkpoints_paths(best_trial, "mean_accuracy"))

        restored_trainable = PytorchTrainable()
        restored_trainable.restore(best_checkpoint[0])
        best_model = restored_trainable.trainer.model
        # Note that test only runs on a small random set of the test data, thus the
        # accuracy may be different from metrics shown in tuning process.
        test_acc = restored_trainable.trainer.test_model()

        return test_acc, best_model

    def construct_network_sizes(self):

        def power_law(input_size, output_size, number_of_layers, layer_number):
            exp = np.log(input_size) - np.log(output_size)
            exp = np.divide(exp, np.log(number_of_layers))
            result = input_size / np.power(layer_number, exp)

            return result

        sizes = {}

        # AutoEncoder
        middle_layers = []

        for i in range(2, self.hidden_encoder + 2):
            current = int(power_law(self.encoder_in, self.core_invariant_size, self.hidden_encoder + 2, i))
            if current <= self.core_invariant_size or current <= 1:
                break
            middle_layers.append(current)

        sizes["encoder"] = [int(self.encoder_in)] + middle_layers + [int(self.core_invariant_size)]

        # Action
        middle_layers = []
        for i in range(2, self.hidden_action + 2):
            current = int(power_law(self.core_invariant_size, self.action_out, self.hidden_action + 2, i))
            if current <= self.core_invariant_size or current <= 1:
                break
            middle_layers.append(current)

        sizes["action"] = [int(self.core_invariant_size)] + middle_layers + [int(self.action_out)]

        return sizes

    def pca_dataset(self, data_loader, threshold=0.96):

        # Most of the time, the datasets are too big to run PCA on it all, so we're going to get a random subset
        # that hopefully will be representative
        print("Doing PCA")
        train, valid, test = data_loader
        train_data = []
        for i, (input, target) in enumerate(train):
            n = input.size()[0]
            indices = np.random.choice(list(range(n)), size=(int(n / 5)))
            input = input.numpy()
            data = input[indices]
            train_data.extend(data)

        train_data = np.array(train_data)
        model = PCA()
        model.fit_transform(train_data)
        variance_cumsum = model.explained_variance_ratio_.cumsum()

        n_comp = 0
        for cum_variance in variance_cumsum:
            if cum_variance >= threshold:
                n_comp += 1
                break
            else:
                n_comp += 1

        print("Done with PCA, got:", n_comp)

        return n_comp
