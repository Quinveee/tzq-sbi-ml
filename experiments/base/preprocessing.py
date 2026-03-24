import sys
from pathlib import Path
import vector
import numpy as np
import torch
import tqdm as tqdm

from models.ParT.ParticleTransformer import ParticleTransformer
from models.MIParT.MIParticleTransformer import MIParticleTransformerWrapper

# Simple wrapper for ParT to match checkpoint structure
class ParTWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)
    
    def forward(self, x, v=None, mask=None):
        return self.mod(x, v=v, mask=mask)

import hydra
from omegaconf import DictConfig, OmegaConf

class BasePreprocessing:
    """
    Base class for preprocessing data. This class can be extended to implement specific preprocessing steps
    for different datasets or models.
    """

    def __init__(self, cfg: DictConfig, key: str) -> None:
        self._key = key
        self.cfg = cfg

    def __call__(self):
        return self.preprocess()

    def __str__(self):
        return self._key if self._key else ""

    def preprocess(self) -> None:
        """
        Preprocess the data from the source location and save the transformed data.

        :param source: location of data folder
        :param model: which model to use for ml features, either "MIParT" or "ParT"
        """
        source = self.cfg.data.source
        model = self.cfg.data.model

        self._convert_data(source, model)


    def _convert_vector(self, E: np.ndarray, px: np.ndarray, py: np.ndarray, pz: np.ndarray):
        return vector.array({'px': px, 'py': py, 'pz': pz, 'energy': E})


    def _reshape_particles(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape the input data to have shape (nsamples, max_particles, 4)
        where max_particles is the maximum number of particles in any event
        and 4 corresponds to the 4 kinematic features (E, px, py, pz)

        :param x: Input data of shape (nsamples, n_features) where n_features is a multiple of 4
        :return: Reshaped data of shape (nsamples, max_particles, 4)
        """
        n_samples, n_features = x.shape
        assert n_features % 4 == 0, "Number of features must be a multiple of 4"
        
        particles = n_features // 4
        return x.reshape(n_samples, particles, 4)


    def _transform(self, data: np.ndarray):
        """
        Transforms the data to pt eta phi using the vector library
        assumes the data is structured according to: E px py pz
        """

        # reshape data to (nsamples, max_particles, 4)
        particles = self._reshape_particles(data)

        E  = np.nan_to_num(particles[:, :, 0],  nan=0.0)
        px = np.nan_to_num(particles[:, :, 1],  nan=0.0)
        py = np.nan_to_num(particles[:, :, 2],  nan=0.0)
        pz = np.nan_to_num(particles[:, :, 3],  nan=0.0)

        # create mask to handle nan values
        mask = ~np.isnan(E)
        def apply_mask(x):
            return np.where(mask, x, np.nan)

        p4 = self._convert_vector(E, px, py, pz)
        jet = p4.sum(axis=1)

        jet_pt  = jet.pt[:, None]
        jet_eta = jet.eta[:, None]
        jet_phi = jet.phi[:, None]
        jet_E   = jet.energy[:, None]

        part_pt  = p4.pt
        part_eta = p4.eta
        part_phi = p4.phi
        part_E   = p4.energy

        d_eta = part_eta - jet_eta
        d_phi = part_phi - jet_phi
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

        dR = np.sqrt(d_eta**2 + d_phi**2)

        eps = 1e-8  # to avoid log(0)

        log_pt = np.log(part_pt + eps)
        log_E  = np.log(part_E + eps)

        log_pt_rel = np.log((part_pt / (jet_pt + eps)) + eps)
        log_E_rel  = np.log((part_E  / (jet_E  + eps)) + eps)

        # apply mask
        part_pt  = apply_mask(part_pt)
        part_eta = apply_mask(part_eta)
        part_phi = apply_mask(part_phi)
        part_E   = apply_mask(part_E)

        d_eta = apply_mask(d_eta)
        d_phi = apply_mask(d_phi)
        dR    = apply_mask(dR)

        log_pt     = apply_mask(log_pt)
        log_E      = apply_mask(log_E)
        log_pt_rel = apply_mask(log_pt_rel)
        log_E_rel  = apply_mask(log_E_rel)

        X = np.stack([
            d_eta,
            d_phi,
            log_pt,
            log_E,
            log_pt_rel,
            log_E_rel,
            dR,
        ], axis=-1)

        # TODO: add more features, e.g. jet mass, missing energy, etc.
        features = np.concatenate([jet_pt, jet_eta, jet_phi, jet_E], axis=-1) # shape (nsamples, 4)

        return X, features


    def _inference_jet_scores(self, model, data: np.ndarray) -> np.ndarray:
        """
        convert the data into batches and pass through the model to get jet scores
        for each event, we take the scores p(q/g), p(W->qq), p(Z->qq) and p(top)
        and add them as features to the data.

        :param model: the model to use for inference
        :param data: the data to pass through the model, shape (nsamples, max_particles, n_features)
        """
        batch_size = (
            OmegaConf.select(self.cfg, "preprocessing.batch_size")
            or OmegaConf.select(self.cfg, "preprocessing.preprocessing.batch_size")
            or 32
        )
        jet_scores = []

        for batch in tqdm.tqdm(self._load_batch(data, batch_size), total=(data.shape[0] // batch_size) + 1, desc="Inferring jet scores"):
            with torch.no_grad():
                batch_tensor = torch.from_numpy(batch).float()
                output = model(batch_tensor)
                jet_scores.append(output.cpu().numpy())

        return np.concatenate(jet_scores, axis=0)



    def _initialize_model(self, model_name: str):
        """
        Initialize the model based on the model name. This function can be extended to support more models.

        :param model_name: Name of the model to initialize (e.g., "MIParT", "ParT")
        :return: Initialized model
        """
        checkpoint_keys = [
            f"preprocessing.models.{model_name}.checkpoint_path",
            f"preprocessing.preprocessing.models.{model_name}.checkpoint_path",
        ]
        checkpoint_value = None
        for checkpoint_key in checkpoint_keys:
            checkpoint_value = OmegaConf.select(self.cfg, checkpoint_key)
            if checkpoint_value is not None:
                break

        if checkpoint_value is None:
            print(
                f"ERROR: Missing config value for checkpoint path. Tried: {checkpoint_keys}"
            )
            return

        checkpoint_path = Path(checkpoint_value)

        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            return
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Infer model architecture from state dict keys
        input_dim = state_dict["mod.embed.input_bn.weight"].numel()
        num_classes = state_dict["mod.fc.0.weight"].shape[0]
        
        if model_name == "MIParT":
            num_mi_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.blocks.")})
            num_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.blocks2.")})
            num_cls_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.cls_blocks.")})
            
            model = MIParticleTransformerWrapper(
                input_dim=input_dim,
                num_classes=num_classes,
                pair_input_dim=4,
                use_pre_activation_pair=False,
                embed_dims=[128, 512, 128],
                pair_embed_dims=[64, 64, 64],
                num_heads=8,
                num_MIlayers=num_mi_layers,
                num_layers=num_layers,
                num_cls_layers=num_cls_layers,
                cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
                fc_params=[],
                activation="gelu",
                trim=True,
                for_inference=True,
                groups=1,
            )
        elif model_name == "ParT":
            num_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.blocks.") and ".blocks." in k})
            num_cls_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.cls_blocks.")})
            
            model = ParTWrapper(
                input_dim=input_dim,
                num_classes=num_classes,
                pair_input_dim=4,
                use_pre_activation_pair=False,
                embed_dims=[128, 512, 128],
                pair_embed_dims=[64, 64, 64],
                num_heads=8,
                num_layers=num_layers,
                num_cls_layers=num_cls_layers,
                cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
                fc_params=[],
                activation="gelu",
                trim=True,
                for_inference=True,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Load state dict and set model to eval mode
        model.load_state_dict(state_dict)
        model.eval()
        return model

    
    def _load_batch(self, data: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Generator function to yield batches of data.

        :param data: Input data of shape (nsamples, max_particles, n_features)
        :param batch_size: Size of each batch
        """
        n_samples = data.shape[0]
        for i in range(0, n_samples, batch_size):
            yield data[i:i + batch_size]


    def _convert_data(self, source: str, model: str = "ParT") -> None:
        """
        Convert data containing 4 kinematics to data with 
        4 kinematics + pt eta phi + engineered features (ML based and 'normal')

        :param source: location of data folder (same as particle experiments)
        :param model: which model to use for ml features, either "MIParT" or "ParT"
        """
        source = Path(source)
        
        # Load train and test data
        print(f"Loading data from {source}")
        x_train = np.load(source / "x_train_ratio.npy")
        x_test = np.load(source / "x_test.npy")
        print(f"Train data shape: {x_train.shape}, Test data shape: {x_test.shape}")

        # Transform both train and test data
        print(f"Transforming data...")
        train_inference_data, train_features = self._transform(x_train)
        test_inference_data, test_features = self._transform(x_test)
        
        # Swap axes for model input (nsamples, n_features, max_particles)
        train_inference_data = train_inference_data.swapaxes(1, 2)
        test_inference_data = test_inference_data.swapaxes(1, 2)

        # Initialize model
        print(f"Initializing {model} model...")
        model_instance = self._initialize_model(model)
        if model_instance is None:
            return

        # Inference jet scores for both
        print(f"Running inference on train data...")
        train_jet_scores = self._inference_jet_scores(model_instance, train_inference_data)
        
        print(f"Running inference on test data...")
        test_jet_scores = self._inference_jet_scores(model_instance, test_inference_data)

        # Concatenate jet scores to features
        train_tf_data = np.concatenate([train_features, train_jet_scores[:, :4]], axis=-1)
        test_tf_data = np.concatenate([test_features, test_jet_scores[:, :4]], axis=-1)

        # Save transformed data
        train_target = source / "x_train_ratio_processed.npy"
        test_target = source / "x_test_processed.npy"
        
        np.save(train_target, train_tf_data)
        np.save(test_target, test_tf_data)
        
        print(f"Transformed train data saved to {train_target}")
        print(f"Transformed test data saved to {test_target}")
