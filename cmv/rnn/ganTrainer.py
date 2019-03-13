from typing import Tuple, List, Iterable, Optional

import torch

from allennlp.training import Trainer

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

class GANTrainer(Trainer):
    #TODO: for now, disallow restoring checkpoints since this can cause a conflict between the discriminator and the generator
     def _restore_checkpoint(self) -> Tuple[int, List[float]]:
         latest_checkpoint = self.find_latest_checkpoint()

         if latest_checkpoint is None:
             return 0, []

         model_path, training_state_path = latest_checkpoint
         training_state = torch.load(training_state_path, map_location=util.device_mapping(-1))

         if isinstance(training_state["epoch"], int):
             epoch_to_return = training_state["epoch"] + 1
         else:
             epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

         if "val_metric_per_epoch" not in training_state:
             logger.warning("trainer state `val_metric_per_epoch` not found, using empty list")
             val_metric_per_epoch: List[float] = []
         else:
             val_metric_per_epoch = training_state["val_metric_per_epoch"]
                 
         return epoch_to_return, val_metric_per_epoch

     # Requires custom from_params.
     @classmethod
     def from_params(cls,
                     model: Model,
                     serialization_dir: str,
                     iterator: DataIterator,
                     train_data: Iterable[Instance],
                     validation_data: Optional[Iterable[Instance]],
                     params: Params,
                     validation_iterator: DataIterator = None) -> 'GANTrainer':

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop_int("cuda_device", -1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                       train_data, validation_data,
                       patience=patience,
                       validation_metric=validation_metric,
                       validation_iterator=validation_iterator,
                       shuffle=shuffle,
                       num_epochs=num_epochs,
                       serialization_dir=serialization_dir,
                       cuda_device=cuda_device,
                       grad_norm=grad_norm,
                       grad_clipping=grad_clipping,
                       learning_rate_scheduler=scheduler,
                       num_serialized_models_to_keep=num_serialized_models_to_keep,
                       keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                       model_save_interval=model_save_interval,
                       summary_interval=summary_interval,
                       histogram_interval=histogram_interval)
