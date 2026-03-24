import sys
from pathlib import Path
import vector
import numpy as np
import torch
import tqdm as tqdm

from models.ParT.ParticleTransformer import ParticleTransformer
from models.MIParT.MIParticleTransformerWrapper import MIParticleTransformerWrapper

import hydra
from omegaconf import DictConfig

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
        batch_size = self.cfg.data.batch_size
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
        if model_name == "MIParT":
            # Initialize MIParT model
            checkpoint_path = Path("/project/atlas/users/qvanenge/code/tzq-sbi-ml/models/MIParT/MIParT_kin.pt")
            if not checkpoint_path.exists():
                print(f"ERROR: Checkpoint not found at {checkpoint_path}")
                return
            
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Infer the key architectural dimensions from the checkpoint
            input_dim = state_dict["mod.embed.input_bn.weight"].numel()
            num_classes = state_dict["mod.fc.0.weight"].shape[0]
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
            # Initialize ParT model
            checkpoint_path = Path("/project/atlas/users/qvanenge/code/tzq-sbi-ml/models/ParT/ParT_kin.pt")
            if not checkpoint_path.exists():
                print(f"ERROR: Checkpoint not found at {checkpoint_path}")
                return
            
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Infer the key architectural dimensions from the checkpoint
            input_dim = state_dict["mod.embed.input_bn.weight"].numel()
            num_classes = state_dict["mod.fc.0.weight"].shape[0]
            num_mi_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.blocks.")})
            num_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.blocks2.")})
            num_cls_layers = len({int(k.split(".")[2]) for k in state_dict if k.startswith("mod.cls_blocks.")})

            model = ParticleTransformer(
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
        Convert data containg 4 kinematics to data with 
        4 kinematics + pt eta phi + engineerd features (ML based and 'normal')

        :param source: location of data folder
        :param model: which model to use for ml features, either "MIParT" or "ParT"
        """

        # load data
        source = Path(source)
        data = np.load(source)

        # transform data
        inference_data, features = self._transform(data) # inference_data shape: (nsamples, max_particles, n_features) feature shape: (nsamples, n_engineered_features)
        inference_data = inference_data.swapaxes(1, 2) # swap to (nsamples, n_features, max_particles) for model input

        # TODO: make model paths configurable in the config file
        # TODO: use initialize_model to load the model instead of hardcoding the paths and model initialization here
        model = self._initialize_model(model)

        # inference jet scores and add to features
        jet_scores = self._inference_jet_scores(model, inference_data) # shape (nsamples, num_classes)

        # concatenate jet scores to features
        tf_data = np.concatenate([features, jet_scores[:, :4]], axis=-1) # shape (nsamples, n_engineered_features + num_classes) where num_classes is typically 4 for q/g, W->qq, Z->qq, top

        # Save transformed data to same location with modified name
        target = source.with_name(source.stem + "_processed" + source.suffix)
        np.savez(target, data=tf_data)
        print(f"Transformed data saved to {target}")
