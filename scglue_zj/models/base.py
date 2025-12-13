r"""
Base classes for model definition and training
"""

import os
import pathlib
import tempfile
from abc import abstractmethod
from typing import Any, Iterable, List, Mapping, Optional
import numpy as np
import sys
import dill
import ignite
import torch
import pandas as pd
# from scglue_zj.models import Model
from scbpi.metrics import avg_silhouette_width

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from ..utils import DelayedKeyboardInterrupt, config, logged

EPOCH_STARTED = ignite.engine.Events.EPOCH_STARTED
EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED
ITERATION_COMPLETED = ignite.engine.Events.ITERATION_COMPLETED
EXCEPTION_RAISED = ignite.engine.Events.EXCEPTION_RAISED
COMPLETED = ignite.engine.Events.COMPLETED
# EPOCH_COMPLETED = ignite.engine.Events.EPOCH_COMPLETED


@logged
class Trainer:

    r"""
    Abstract trainer class

    Parameters
    ----------
    net
        Network module to be trained

    Note
    ----
    Subclasses should populate ``required_losses``, and additionally
    define optimizers here.
    """

    def __init__(self,
                 # model:Model ,
                 net: torch.nn.Module) -> None:
        self.net = net
        # self.model = model
        self.required_losses: List[str] = []

    @abstractmethod
    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        r"""
        A single training step

        Parameters
        ----------
        engine
            Training engine
        data
            Data of the training step

        Returns
        -------
        loss_dict
            Dict containing training loss values
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def val_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        r"""
        A single validation step

        Parameters
        ----------
        engine
            Validation engine
        data
            Data of the validation step

        Returns
        -------
        loss_dict
            Dict containing validation loss values
        """
        raise NotImplementedError  # pragma: no cover

    def report_metrics(
            self, train_state: ignite.engine.State,
            val_state: Optional[ignite.engine.State]
    ) -> None:
        r"""
        Report loss values during training

        Parameters
        ----------
        train_state
            Training engine state
        val_state
            Validation engine state
        """
        if train_state.epoch % config.PRINT_LOSS_INTERVAL:
            return
        train_metrics = {
            key: float(f"{val:.3f}")
            for key, val in train_state.metrics.items()
        }
        val_metrics = {
            key: float(f"{val:.3f}")
            for key, val in val_state.metrics.items()
        } if val_state else None
        self.logger.info(
            "[Epoch %d] train=%s, val=%s, %.1fs elapsed",
            train_state.epoch, train_metrics, val_metrics,
            train_state.times["EPOCH_COMPLETED"]  # Also includes validator time
        )

    # def _on_epoch_completed(self, engine):
    #     print("0")
        # if not self.Fine_tuning:  # 如果 Fine_tuning 不为 True, 直接返回
        #     return
        # print("1")
        # tol = 1e-3
        # epoch = engine.state.epoch
        # # 将 z_embedding_list 中的所有 z_samp_joint 拼接成 z_embedding
        # z_embedding = torch.cat(self.z_embedding_list, dim=0)
        # self.z_embedding_list = []  # 清空 z_embedding_list，为下个 epoch 做准备
        #
        # # 执行 kmeans_loss 计算并保存距离
        # dist, _ = self.kmeans_loss(z_embedding)
        # # np.savetxt("GSE126074_embedding.csv", z_embedding.numpy(), delimiter=",")
        #
        # # 更新 y_pred
        # self.y_pred = torch.argmin(dist, dim=1).cpu().numpy()
        #
        # print(f"Epoch {epoch} completed, k-means loss computed.")
        # # check stop criterion
        # delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / z_embedding.shape[0]
        # self.y_pred_last = self.y_pred
        # if epoch > 0 and delta_label < tol:
        #     print('delta_label ', delta_label, '< tol ', tol)
        #     print("Reach tolerance threshold. Stopping training.")
        #     engine.terminate() # 结束整个程序

    def fit(
            self, train_loader: Iterable, val_loader: Optional[Iterable] = None,
            max_epochs: int = 100, random_seed: int = 0,
            directory: Optional[os.PathLike] = None,
            plugins: Optional[List["TrainingPlugin"]] = None
    ) -> None:
        r"""
        Fit network

        Parameters
        ----------
        train_loader
            Training data loader
        val_loader
            Validation data loader
        max_epochs
            Maximal number of epochs
        random_seed
            Random seed
        directory
            Training directory
        plugins
            Optional list of training plugins
        """
        interrupt_delayer = DelayedKeyboardInterrupt()
        directory = pathlib.Path(directory or tempfile.mkdtemp(prefix=config.TMP_PREFIX))
        self.logger.info("Using training directory: \"%s\"", directory)

        # Construct engines
        train_engine = ignite.engine.Engine(self.train_step)
        val_engine = ignite.engine.Engine(self.val_step) if val_loader else None

        # @train_engine.on(EPOCH_COMPLETED)
        # def on_epoch_completed(engine):
        #     if not self.Fine_tuning:  # 如果 Fine_tuning 不为 True, 直接返回
        #         return
        #     # print("1")
        #     tol = 1e-3
        #     epoch = engine.state.epoch
        #     # 将 z_embedding_list 中的所有 z_samp_joint 拼接成 z_embedding
        #     # z_embedding_ = torch.cat(self.z_embedding_list, dim=0)
        #     # self.z_embedding_list = []  # 清空 z_embedding_list，为下个 epoch 做准备
        #     from .scglue import SCGLUEModel
        #
        #     adatas = self.adatas
        #     z_embedding = self.encode_zembedding()
        #     # # 执行 kmeans_loss 计算并保存距离
        #     dist, _ = self.kmeans_loss(z_embedding)
        #     # np.savetxt("GSE126074_embedding.csv", z_embedding.numpy(), delimiter=",")
        #     # for k, adata in adatas.items():
        #     #     if adata is 'rna':
        #     #         true_labels = adata.obs['cell_type']
        #     # 假设 adats 是一个包含 'rna' 的字典
        #     rna = adatas['rna']
        #     # 提取 rna.obs 中的 'cell_type' 列为 true_labels
        #     true_labels = rna.obs['cell_type']
        #     # 更新 y_pred
        #     # self.y_pred = torch.argmin(dist, dim=1).cpu().numpy()
        #     true_labels = pd.Series(true_labels.values, index=true_labels.index)
        #     ami = np.round(adjusted_mutual_info_score(true_labels, self.y_pred), 5)
        #     nmi = np.round(normalized_mutual_info_score(true_labels, self.y_pred), 5)
        #     ari = np.round(adjusted_rand_score(true_labels, self.y_pred), 5)
        #     Silhouette_score = np.round(avg_silhouette_width(z_embedding, true_labels), 5)
        #     # final_epoch = epoch + 1
        #     print('Clustering   %d: AMI= %.4f, NMI= %.4f, ARI= %.4f, Silhouette= %.4f' % (epoch, ami, nmi, ari, Silhouette_score))
        #     # print(f"Epoch {epoch} completed, k-means loss computed.")
        #     # check stop criterion
        #     delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / z_embedding.shape[0]
        #     print('delta_label :', delta_label)
        #     self.y_pred_last = self.y_pred
        #     if epoch > 0 and delta_label < tol:
        #         print('delta_label ', delta_label, '< tol ', tol)
        #         print("Reach tolerance threshold. Stopping training.")
        #         engine.terminate()  # 结束整个程序

        delay_interrupt = interrupt_delayer.__enter__
        train_engine.add_event_handler(EPOCH_STARTED, delay_interrupt)
        train_engine.add_event_handler(COMPLETED, delay_interrupt)
        # Exception handling
        train_engine.add_event_handler(ITERATION_COMPLETED, ignite.handlers.TerminateOnNan())
        # train_engine.add_event_handler(EPOCH_COMPLETED, self.on_epoch_completed)
        @train_engine.on(EXCEPTION_RAISED)
        def _handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and config.ALLOW_TRAINING_INTERRUPTION:
                self.logger.info("Stopping training due to user interrupt...")
                engine.terminate()
            else:
                raise e

        # Compute metrics
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(train_engine, item)
            if val_engine:
                ignite.metrics.Average(
                    output_transform=lambda output, item=item: output[item]
                ).attach(val_engine, item)

        if val_engine:
            @train_engine.on(EPOCH_COMPLETED)
            def _validate(engine):
                val_engine.run(
                    val_loader, max_epochs=engine.state.epoch
                )  # Bumps max_epochs by 1 per training epoch, so validator resumes for 1 epoch

        @train_engine.on(EPOCH_COMPLETED)
        def _report_metrics(engine):
            self.report_metrics(engine.state, val_engine.state if val_engine else None)

        # @train_engine.on(EPOCH_COMPLETED)
        # def _on_epoch_completed(engine):
        #     self.on_epoch_completed(engine)

        for plugin in plugins or []:
            plugin.attach(
                net=self.net, trainer=self,
                train_engine=train_engine, val_engine=val_engine,
                train_loader=train_loader, val_loader=val_loader,
                directory=directory
            )

        restore_interrupt = lambda: interrupt_delayer.__exit__(None, None, None)
        # train_engine.add_event_handler(EPOCH_COMPLETED, self.on_epoch_completed(train_engine))
        train_engine.add_event_handler(EPOCH_COMPLETED, restore_interrupt)
        train_engine.add_event_handler(COMPLETED, restore_interrupt)

        # Start engines
        torch.manual_seed(random_seed)
        train_engine.run(train_loader, max_epochs=max_epochs)

        torch.cuda.empty_cache()  # Works even if GPU is unavailable

    def get_losses(self, loader: Iterable) -> Mapping[str, float]:
        r"""
        Get loss values for given data

        Parameters
        ----------
        loader
            Data loader

        Returns
        -------
        loss_dict
            Dict containing loss values
        """
        engine = ignite.engine.Engine(self.val_step)
        for item in self.required_losses:
            ignite.metrics.Average(
                output_transform=lambda output, item=item: output[item]
            ).attach(engine, item)
        engine.run(loader, max_epochs=1)
        torch.cuda.empty_cache()  # Works even if GPU is unavailable
        return engine.state.metrics

    def state_dict(self) -> Mapping[str, Any]:
        r"""
        State dict

        Returns
        -------
        state_dict
            State dict
        """
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        r"""
        Load state from a state dict

        Parameters
        ----------
        state_dict
            State dict
        """


@logged
class Model:

    r"""
    Abstract model class

    Parameters
    ----------
    net
        Network type
    *args
        Positional arguments are passed to the network constructor
    **kwargs
        Keyword arguments are passed to the network constructor

    Note
    ----
    Subclasses may override arguments for API definition.
    """

    NET_TYPE = torch.nn.Module
    TRAINER_TYPE = Trainer

    def __init__(self, *args, **kwargs) -> None:
        self._net = self.NET_TYPE(*args, **kwargs)
        self._trainer: Optional[Trainer] = None  # Constructed upon compile

    @property
    def net(self) -> torch.nn.Module:
        r"""
        Neural network module in the model (read-only)
        """
        return self._net

    @property
    def trainer(self) -> Trainer:
        r"""
        Trainer of the neural network module (read-only)
        """
        if self._trainer is None:
            raise RuntimeError(
                "No trainer has been registered! "
                "Please call `.compile()` first."
            )
        return self._trainer

    def compile(self, *args, **kwargs) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        trainer
            Trainer type
        *args
            Positional arguments are passed to the trainer constructor
        **kwargs
            Keyword arguments are passed to the trainer constructor

        Note
        ----
        Subclasses may override arguments for API definition.
        """
        if self._trainer:
            self.logger.warning(
                "`compile` has already been called. "
                "Previous trainer will be overwritten!"
            )
        self._trainer = self.TRAINER_TYPE(self.net, *args, **kwargs)

    def fit(self, *args, **kwargs) -> None:
        r"""
        Alias of ``.trainer.fit``.

        Parameters
        ----------
        *args
            Positional arguments are passed to the ``.trainer.fit`` method
        **kwargs
            Keyword arguments are passed to the ``.trainer.fit`` method

        Note
        ----
        Subclasses may override arguments for API definition.
        """
        self.trainer.fit(*args, **kwargs)

    def get_losses(self, *args, **kwargs) -> Mapping[str, float]:
        r"""
        Alias of ``.trainer.get_losses``.

        Parameters
        ----------
        *args
            Positional arguments are passed to the ``.trainer.get_losses`` method
        **kwargs
            Keyword arguments are passed to the ``.trainer.get_losses`` method

        Returns
        -------
        loss_dict
            Dict containing loss values
        """
        return self.trainer.get_losses(*args, **kwargs)

    def save(self, fname: os.PathLike) -> None:
        r"""
        Save model to file

        Parameters
        ----------
        file
            Specifies path to the file

        Note
        ----
        Only the network is saved but not the trainer
        """
        fname = pathlib.Path(fname)
        trainer_backup, self._trainer = self._trainer, None
        device_backup, self.net.device = self.net.device, torch.device("cpu")
        with fname.open("wb") as f:
            dill.dump(self, f, protocol=4, byref=False, recurse=True)
        self.net.device = device_backup
        self._trainer = trainer_backup

    def upgrade(self) -> None:
        r"""
        Upgrade the model if generated by older versions
        """

    # def encode_graph(self, guidance_hvf):
    #     return self.encode_graph()


@logged
class TrainingPlugin:

    r"""
    Plugin used to extend the training process with certain functions
    """

    @abstractmethod
    def attach(
            self, net: torch.nn.Module, trainer: Trainer,
            train_engine: ignite.engine.Engine,
            val_engine: ignite.engine.Engine,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            directory: pathlib.Path
    ) -> None:
        r"""
        Attach custom handlers to training or validation engine

        Parameters
        ----------
        net
            Network module
        trainer
            Trainer object
        train_engine
            Training engine
        val_engine
            Validation engine
        train_loader
            Training data loader
        val_loader
            Validation data loader
        directory
            Training directory
        """
        raise NotImplementedError  # pragma: no cover
