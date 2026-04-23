"""
Base experiment class for all ML-based experiments
"""

from __future__ import annotations

import copy
import random
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..logger import LOGGER as _LOGGER
from ..plotting import plot_learning_curves
from ..utils import device, dtype
from .base_experiment import BaseExperiment
from .normalizers import Normalizer, get_normalizer
from .schemas import Losses

if TYPE_CHECKING:

    from ..losses import Loss
    from .schemas import Limits

LOGGER = _LOGGER.getChild(__name__)


@contextmanager
def needs_grad(flag):
    ctx = torch.enable_grad() if flag else torch.no_grad()
    with ctx:
        yield


class BaseExperimentML(BaseExperiment):
    """
    Base experiment class for all ML experiments
    The class level attributes `collate_fn`, `dataset_cls` and `asymptotics_cls`
    are to be defined by subclasses
    """

    collate_fn = None
    dataset_cls = None
    asymptotics_cls = None

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.model = None
        self.normalizer = None
        self.score_normalizer: Optional[Normalizer] = None
        self.loss_fn: Optional[Loss] = None
        self.train_loader = self.val_loader = None
        self.train_dataset = self.val_dataset = None
        self.test_dataset = self.test_loader = None
        self.device_kwds = {
            "device": device(self.cfg.devices.device),
            "dtype": dtype(self.cfg.devices.dtype),
            "non_blocking": self.cfg.devices.non_blocking,
        }
        self._seed: Optional[int] = None
        self._generator: Optional[torch.Generator] = None

    ## Following abstract methods are to be implemented by subclasses ##

    @abstractmethod
    def _load_raw_data(self, *args, **kwds) -> ...: ...

    @abstractmethod
    def _load_dataset(self, raw, mode: Literal["train", "test"] = "train") -> ...: ...

    @abstractmethod
    def _preds(self, batch) -> ...: ...

    @abstractmethod
    def _eval(self, output) -> ...: ...

    ##

    def _init(self) -> None:
        """
        Datasets must be initialized before loaders.
        The order of the resting methods is irrelevant

        """
        self.init_seed()
        self.init_loss_function()
        self.init_model()
        self.init_normalizer()
        self.init_datasets()
        self.init_loaders()

    def init_seed(self) -> None:
        """
        Seed all RNGs (python, numpy, torch CPU/CUDA) from ``cfg.data.run`` so
        weight init, train/val split and shuffle order are reproducible across
        runs and identical between runs that share the same ``data.run`` value.
        A non-integer ``data.run`` (e.g. wandb run id) is hashed deterministically.
        """
        raw = self.cfg.data.get("run", 0)
        try:
            seed = int(raw)
        except (TypeError, ValueError):
            seed = abs(hash(str(raw))) % (2**32)
        self._seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        LOGGER.info(f"Set RNG seed to {seed} (data.run={raw})")

    def _run(self) -> None:
        """
        Run training, evaluation and plotting

        """
        if self.cfg.modes.train:
            LOGGER.info(f"Starting traing of exp. {self}")
            self.checkpoints.state_dict, self.checkpoints.losses = self.train()
        if self.cfg.modes.eval:
            LOGGER.info(f"Starting evaluation of exp. {self}")
            self.checkpoints.limits = self.eval_lims()
        if self.cfg.modes.plot:
            LOGGER.info(f"Plotting results of experiment {self}")
            self.plot()

    def _resolve_run_model_key(self) -> str:
        """Resolve model key used for run naming and run-dir conventions."""
        run_model_key = self.cfg.data.get("run_model_key", None)
        if run_model_key:
            return run_model_key

        model_key = self.cfg.model.get("key", "")
        if model_key == "transformer":
            lloca_cfg = self.cfg.model.get("LLoCa", {})
            if lloca_cfg.get("active", False):
                return "transformer_lloca"

        return model_key

    def init_loss_function(self) -> None:
        """
        Instantiate loss from configuration object

        """
        self.loss_fn = instantiate(self.cfg.loss)

    def init_datasets(self) -> None:
        """
        Load raw data, normalize, create splits and store it in
        torch datasets

        """
        # Load raw data
        raw = self._load_raw_data(self.cfg.dataset.path)

        # Normalize raw
        assert self.normalizer is not None
        raw.x_train = self.normalizer.fit_transform(raw.x_train)
        raw.x_test = self.normalizer.transform(raw.x_test)
        n_mask = getattr(self.normalizer, "n_mask_cols", 0)
        if n_mask:
            LOGGER.info(
                "Normalizer appended %d NaN-presence mask columns; "
                "post-normalization x dim = %d",
                n_mask,
                raw.x_train.shape[-1],
            )

        # Create datasets
        dataset = self._load_dataset(raw, "train")

        split = self.cfg.train.validation_split
        assert 0.0 < split < 0.5, f"Validation split {split} seems wrong"

        # Split manually (use seeded generator so split is reproducible per run)
        split_gen = (
            torch.Generator().manual_seed(self._seed)
            if self._seed is not None
            else None
        )
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [
                int(len(dataset) * (1 - split)),
                int(len(dataset) * split),
            ],
            generator=split_gen,
        )
        self.test_dataset = self._load_dataset(raw, "test")

    def init_loaders(self) -> None:
        """
        Create torch loaders for the different splits

        """
        loader_gen = (
            torch.Generator().manual_seed(self._seed)
            if self._seed is not None
            else None
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            pin_memory=self.cfg.devices.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=0,
            generator=loader_gen,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,   # no need to shuffle validation data
            pin_memory=self.cfg.devices.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,   # no need to shuffle test data
            pin_memory=self.cfg.devices.pin_memory,
            collate_fn=self.collate_fn,
            num_workers=0,
        )

    def init_model(self) -> None:
        """
        Initialize model from configuration object
        If specified, update model state dict from the checkpoints file
        Move model to appropriate device

        """
        # Initialize model
        self.model = instantiate(self.cfg.model)

        have_ckpt = self.checkpoints.state_dict is not None
        recycle = bool(self.cfg.modes.recycle)

        if not self.cfg.modes.train and have_ckpt and not recycle:
            LOGGER.warning(
                "modes.train=false with an existing checkpoint but modes.recycle=false. "
                "Auto-enabling recycle so eval uses the saved weights, not random ones."
            )
            recycle = True

        if recycle and have_ckpt:
            LOGGER.info("Starting warm!")
            self.model.load_state_dict(self.checkpoints.state_dict)
        elif not self.cfg.modes.train:
            LOGGER.warning(
                "No training this run and no checkpoint to load — evaluation will run "
                "on randomly-initialized weights and its output is meaningless."
            )

        # Move model to device
        self.model = self.model.to(**self.device_kwds)

        LOGGER.info(f"Model moved to {str(self.device_kwds["device"])}")

    def init_normalizer(self) -> None:
        """Set normalizer object based on model type"""
        self.normalizer = get_normalizer(str(self.model))

    def loss(self, batch):
        """
        Just pass the batch to the loss function (set by subclasses)

        """
        return self.loss_fn(self._preds(batch))

    def train(self) -> Tuple[Dict, Losses]:
        """
        Training loop

        :return: Return model state dict and losses
        :rtype: Tuple[Dict, Losses]
        """
        train_losses, val_losses = [], []
        opt = torch.optim.AdamW(params=self.model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

        # Cosine LR with warm restarts; optionally preceded by linear warmup.
        # - lr_restart_period (epochs): T_0, the first restart period. When unset
        #   (or 0), defaults to (total_steps - warmup_steps), so with T_mult=1
        #   the schedule reduces to a single cosine cycle (previous behavior).
        # - lr_restart_mult: T_mult, multiplies the period after each restart.
        total_steps = self.cfg.train.epochs * len(self.train_loader)
        warmup_epochs = int(self.cfg.train.get("lr_warmup", 0) or 0)
        warmup_steps = max(1, warmup_epochs * len(self.train_loader)) if warmup_epochs > 0 else 0

        restart_epochs = int(self.cfg.train.get("lr_restart_period", 0) or 0)
        if restart_epochs > 0:
            T_0 = max(1, restart_epochs * len(self.train_loader))
        else:
            T_0 = max(1, total_steps - warmup_steps)
        T_mult = int(self.cfg.train.get("lr_restart_mult", 1) or 1)

        lr_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=self.cfg.train.get("lr_min", self.cfg.train.lr * 1e-3),
        )

        if warmup_steps > 0:
            lr_warmup = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1.0 / warmup_steps,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            lr_sch = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[lr_warmup, lr_cosine],
                milestones=[warmup_steps],
            )
            LOGGER.info(
                "Using linear LR warmup for %d epochs (%d steps), then cosine warm restarts (T_0=%d steps, T_mult=%d)",
                warmup_epochs,
                warmup_steps,
                T_0,
                T_mult,
            )
        else:
            lr_sch = lr_cosine
            LOGGER.info(
                "Using cosine LR with warm restarts (T_0=%d steps, T_mult=%d, no warmup)",
                T_0,
                T_mult,
            )

        # TensorBoard
        writer = SummaryWriter(log_dir="runs/model_experiment")

        # Weights & Biases
        use_wandb = self.cfg.modes.get("wandb", False)
        created_wandb_run = False
        if use_wandb:
            run_model_key = self._resolve_run_model_key()
            run_name = (
                f"{self.cfg.dataset.key}/{self.cfg.exp.key}/{run_model_key}/run{self.cfg.data.run}"
            )
            if wandb.run is None:
                wandb.login()  # reads WANDB_API_KEY env var automatically
                wandb.init(
                    project="tzq-sbi-ml",
                    name=run_name,
                    config=OmegaConf.to_container(self.cfg, resolve=False),
                    dir="runs/",
                )
                created_wandb_run = True
            else:
                wandb.config.update(
                    OmegaConf.to_container(self.cfg, resolve=False),
                    allow_val_change=True,
                )
                if not wandb.run.name:
                    wandb.run.name = run_name
            # Init lazy modules with one dummy batch through experiment-specific preds.
            dummy = next(iter(self.train_loader))
            dummy.to_(**self.device_kwds)
            self._preds(dummy)
                
            wandb.summary["n_parameters"] = sum(p.numel() for p in self.model.parameters())

        # Training loop
        global_step = 0
        best_val_loss = float("inf")
        best_state_dict = None
        best_epoch = -1

        for e in range(self.cfg.train.epochs):
            self.model.train()
            train_loss = 0.0
            n_valid_batches = 0

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {e+1}/{self.cfg.train.epochs}",
                leave=False,
            )
            for batch in pbar:
                # Zero grads
                opt.zero_grad()

                # Move data to device and cast to dtype
                batch.to_(**self.device_kwds)

                # Calculate loss on output
                loss = self.loss(batch)

                # check loss for NaNs or infs before backward pass
                if not torch.isfinite(loss):
                    LOGGER.warning(
                        f"Non-finite loss detected at epoch {e+1}, global step {global_step}. Skipping backward pass and optimizer step for this batch."
                    )
                    opt.zero_grad()  # Clear any existing gradients
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping (can be `null` or not appear in config)
                max_norm = self.cfg.train.get("clip_grad_norm", float("inf"))
                max_norm = max_norm if max_norm is not None else float("inf")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_norm,
                    error_if_nonfinite=False
                )

                # Optimizer step
                opt.step()
                if lr_sch is not None:
                    lr_sch.step()

                train_loss += loss.item()
                n_valid_batches += 1
                global_step += 1

                # log batch loss to TensorBoard and wandb
                writer.add_scalar("Loss/batch", loss.item(), global_step)
                writer.add_scalar("LR", opt.param_groups[0]["lr"], global_step)
                if use_wandb:
                    wandb.log({"loss/batch": loss.item(), "lr": opt.param_groups[0]["lr"]}, step=global_step)

                # Update progress bar
                pbar.set_postfix(
                    {"batch_loss": loss.item(), "lr": opt.param_groups[0]["lr"]}
                )

            avg_train_loss = train_loss / max(1, n_valid_batches)
            print(
                f"Epoch {e+1}/{self.cfg.train.epochs} - train loss: {avg_train_loss:.4f}"
                + (f"  ({len(self.train_loader) - n_valid_batches} batches skipped)"
                   if n_valid_batches < len(self.train_loader) else "")
            )

            # log epoch loss to TensorBoard and wandb
            writer.add_scalar("Loss/train_epoch", avg_train_loss, e + 1)
            if use_wandb:
                wandb.log({"loss/train_epoch": avg_train_loss}, step=global_step)
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0

            # If the loss function requires computing the score as the gradient
            # of the output w.r.t the input parameters, we need also a graph for
            # the evaluation
            with needs_grad(self.loss_fn.REQUIRES_SCORE):
                for batch in self.val_loader:
                    batch.to_(**self.device_kwds)
                    val_loss += self.loss(batch).item()

            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {e+1}/{self.cfg.train.epochs} - val loss: {avg_val_loss:.4f}")
            writer.add_scalar("Loss/val_epoch", avg_val_loss, e + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = e + 1
                best_state_dict = copy.deepcopy(self.model.state_dict())
                LOGGER.info(
                    f"New best val loss {best_val_loss:.4f} at epoch {best_epoch}"
                )

            if use_wandb:
                wandb.log(
                    {
                        "loss/val_epoch": avg_val_loss,
                        "loss/val_best": best_val_loss,
                        "loss/val_best_epoch": best_epoch,
                    },
                    step=global_step,
                )
            val_losses.append(avg_val_loss)

        writer.close()
        if use_wandb and created_wandb_run:
            wandb.finish()

        final_state_dict = best_state_dict if best_state_dict is not None else self.model.state_dict()
        if best_state_dict is not None:
            LOGGER.info(
                f"Returning best checkpoint from epoch {best_epoch} (val loss {best_val_loss:.4f})"
            )
            self.model.load_state_dict(best_state_dict)
        return final_state_dict, Losses(train=train_losses, val=val_losses)

    @torch.no_grad()
    def eval(self, loader: DataLoader) -> np.ndarray:
        """
        Evaluate loader and return predictions as a numpy array

        :param loader: Data laoder
        :type loader: DataLoader
        :return: Predictions as a numpy array
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        self.model.eval()
        device_kwds = self.device_kwds.copy()
        device_kwds["device"] = device(self.cfg.devices.eval)
        self.model = self.model.to(**device_kwds)
        LOGGER.info(f"Model moved to {device_kwds["device"]}")

        preds = []

        LOGGER.info(
            f"Starting to evaluate {len(loader)} minibatches of size {loader.batch_size}"
        )
        for batch in tqdm(loader):
            batch.to_(**device_kwds)
            output = self._preds(batch)
            preds.append(
                self._eval(output).detach()
            )  # i keep detach for the possible future...

        return torch.cat([p.cpu() for p in preds]).numpy()

    def eval_lims(self) -> Limits:
        """
        Calculate asymptotic limits
        If using score regressors, first we create histograms of the score
        If specified in the config, create an Asimov dataset using the test partition
        Return the calculated limits object

        ::note:: This method relies heaviliy on how things are implemented in
        Madminer and will be substantially modified in the future

        :return: Limits object
        :rtype: Limits
        """
        events_file = self.cfg.dataset.events_file
        alims = self.asymptotics_cls(events_file)

        assert (
            len(self.cfg.limits.theta_ranges)
            == len(self.cfg.limits.resolutions)
            == self.cfg.dataset.theta_dim
        ), f"Shape mismatch for dim(theta)={self.cfg.dataset.theta_dim}"

        # Create theta grid
        theta_grid = alims.theta_grid(
            theta_ranges=self.cfg.limits.theta_ranges,
            resolutions=self.cfg.limits.resolutions,
        )

        # Load test data

        # If asimov is available, sample asimov
        if self.cfg.limits.get("asimov"):
            LOGGER.info(
                f"Starting sampling Asimov dataset from test partition of {events_file}"
            )

            assert (
                len(self.cfg.limits.asimov.theta_true) == self.cfg.dataset.theta_dim
            ), "Shape mismatch"

            # ensures results dont differ when using eval only
            if self._seed is not None:
                np.random.seed(self._seed)

            # Sample weighted events from partition dataset
            # `cfg.limits.test_split` defines where the test partition starts
            x_test, weights_test = alims.asimov_data(
                self.cfg.limits.asimov.theta_true,
                self.cfg.limits.asimov.sample_only_from_closest_benchmark,
                self.cfg.limits.test_split,
                self.cfg.limits.asimov.n_asimov,
            )

            # Expected number of events (for *rate* llr estimation)
            n_events = (
                self.cfg.limits.luminosity
                # NOTE: first parameter needs to be a list of arrays or list of lists
                * alims.calculate_xsecs(
                    [self.cfg.limits.asimov.theta_true], self.cfg.limits.test_split
                )[0]
            )

            LOGGER.info(f"Sampled {len(x_test)} events")

            # Put sampled asimov events into a torch loader for evaluation
            # In case of histos, there will be one loader for each point in the
            # parameter grid
            test_loaders = self.create_lims_loaders(
                x=x_test,
                theta=theta_grid if not self.asymptotics_cls.NEEDS_HISTOS else None,
            )

        # If asimov is not available use mc toy
        else:
            weights_test = np.ones(len(self.test_dataset))
            n_events = len(self.test_dataset)
            test_loaders = [self.test_loader]

        LOGGER.info("Evaluating test data ...")
        preds = [self.eval(tl) for tl in test_loaders]

        # Histos if needed
        histos = None
        if self.asymptotics_cls.NEEDS_HISTOS:
            # We need *weighted* events for histogram creation, so we
            # resample from the training partition
            LOGGER.info(f"Sampling train samples from train partition of {events_file}")
            # Same seeding rationale as the Asimov branch above: madminer
            # uses the global np.random for event resampling.
            if self._seed is not None:
                np.random.seed(self._seed)
            x_train, weights_train, _ = alims.weighted_events_from_partition(
                n_draws=self.cfg.limits.n_toys,
                partition="train",
                test_split=self.cfg.limits.test_split,
                thetas=theta_grid,
            )
            x_train = self.normalizer.transform(x_train)
            LOGGER.info(f"Sampled {len(x_train)} train events")
            train_dataset = self.dataset_cls(
                x=x_train,
                theta=theta_grid,
                met=getattr(self, "_use_met", False),
            )
            train_loader = DataLoader(
                train_dataset, batch_size=128, collate_fn=self.collate_fn
            )
            LOGGER.info("Evaluating train data for histograms creation ...")
            preds_train = self.eval(train_loader)
            histos = alims.histos(preds_train, weights_train)

        return alims.limits(
            predictions=preds,
            n_events=n_events,
            x_weights=weights_test,
            theta_grid=theta_grid,
            luminosity=self.cfg.limits.luminosity,
            test_split=self.cfg.limits.test_split,
            histos=histos,
        )

    def create_lims_loaders(self, x, theta=None):
        """
        Method to create torch loaders when using Asimov
        """
        factory_kwds = {"batch_size": 128, "collate_fn": self.collate_fn}
        assert self.dataset_cls is not None and self.normalizer is not None
        x = self.normalizer.transform(x)
        ds_kwds = {"met": getattr(self, "_use_met", False)}
        if theta is None:
            return [DataLoader(self.dataset_cls(x=x, **ds_kwds), **factory_kwds)]
        return [
            DataLoader(
                self.dataset_cls(
                    x=x,
                    # `copy()` is needed to create a writable array, not only a view
                    theta=np.broadcast_to(t, (x.shape[0], t.shape[-1])).copy(),
                    **ds_kwds,
                ),
                **factory_kwds,
            )
            for t in theta
        ]

    def plot_learning_curves(self, to=None):
        assert self.checkpoints is not None
        return plot_learning_curves(self.checkpoints.losses, to=to)

    def plot(self) -> None:
        super().plot(str(self.model))
        self.plot_learning_curves(Path(self.cfg.data.run_dir) / "learning_curve.png")
