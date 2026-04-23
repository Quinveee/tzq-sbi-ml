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
        met = self.cfg.data.met

        self._convert_data(source, model, met=met)


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


    def _transform(self, data: np.ndarray, met: bool = False) -> np.ndarray:
        """
        Transforms the data to pt eta phi using the vector library
        assumes the data is structured according to: E px py pz
        """
        
        # Handle optional MET features
        if met:
            met_features = data[:, -2:]  # Assuming MET features are the last 2 features
            data = data[:, :-2]  # Remove MET features from particle data
            met_pt = met_features[:, 0:1]
            met_phi = met_features[:, 1:2]

            met_px = met_pt * np.cos(met_phi)
            met_py = met_pt * np.sin(met_phi)


        # Reshape data to (nsamples, max_particles, 4)
        particles = self._reshape_particles(data)

        E  = np.nan_to_num(particles[:, :, 0],  nan=0.0)
        px = np.nan_to_num(particles[:, :, 1],  nan=0.0)
        py = np.nan_to_num(particles[:, :, 2],  nan=0.0)
        pz = np.nan_to_num(particles[:, :, 3],  nan=0.0)

        # Create mask to handle nan values
        mask = ~np.isnan(E)
        def apply_mask(x):
            return np.where(mask, x, np.nan)

        # Build vector array and calculate jet 4-momentum by summing over particles
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

        # Calculate relative features
        d_eta = part_eta - jet_eta
        d_phi = part_phi - jet_phi
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

        dR = np.sqrt(d_eta**2 + d_phi**2)

        eps = 1e-8  # to avoid log(0)

        log_pt = np.log(part_pt + eps)
        log_E  = np.log(part_E + eps)

        log_pt_rel = np.log((part_pt / (jet_pt + eps)) + eps)
        log_E_rel  = np.log((part_E  / (jet_E  + eps)) + eps)


        # # Nessacery for high level features
        # bjet = ...          # jet with highest b-tag
        # forward_jet = ...   # highest |eta|
        # radiation_jet = ... # non-b jet, not forward

        # l1, l2 = leptons[0], leptons[1]
        # p_Z = fourvec(l1) + fourvec(l2)

        # # High level features:
        # m_bjf = invariant_mass(fourvec(bjet) + fourvec(forward_jet)) # Invariant mass of the b-jet and the forward jet
        # m_top = invariant_mass(p_top) # Reconstructed top-quark mass
        # m_ll = invariant_mass(p_Z) # Reconstructed Z boson mass
        # m_jj = invariant_mass(fourvec(jets[0]) + fourvec(jets[1])) # Invariant mass of the two leading jets
        # m_bj = invariant_mass(fourvec(bjet) + fourvec(jets[0])) # Invariant mass of the b-jet and the leading jet
        # # Invariant mass of all selected particles in the event
        # # all visible objects
        # p_all = np.zeros(4)
        # for obj in leptons + jets:
        #     p_all += fourvec(obj)

        # m_all = invariant_mass(p_all)

        # eta_fj = abs(eta(forward_jet.px, forward_jet.py, forward_jet.pz)) # Absolute value of the forward jet η
        # eta_j_rad = abs(eta(radiation_jet.px, radiation_jet.py, radiation_jet.pz)) # Absolute value of the radiation jet η
        # btag_score = bjet.btag # b-tagging score of the b-jet
        # deltaR_fjZ = ... # ΔR between the forward jet and the reconstructed Z boson

        # pt_Z = pt(p_Z[1], p_Z[2]) # Transverse momentum of the reconstructed Z boson
        # pt_W = pt(p_W[1], p_W[2]) # Transverse momentum of the reconstructed W boson
        # pt_top = pt(p_top[1], p_top[2]) # Transverse momentum of the reconstructed top quark
        # h_t = part_pt.sum(axis=1, keepdims=True) # Scalar sum of the pT of the selected particles in the event

        # if met:
        #     # Calculate transverse mass of the W boson using jet and MET information
        #     met_mW = np.sqrt(2 * jet_pt * met_pt * (1 - np.cos(jet_phi - met_phi)))

        # Invariant mass of all particles in the event, calculated from the sum of their 4-momenta
        p_all_E  = np.sum(E, axis=1)
        p_all_px = np.sum(px, axis=1)
        p_all_py = np.sum(py, axis=1)
        p_all_pz = np.sum(pz, axis=1)

        m_all = np.sqrt(np.maximum(
            p_all_E**2 - (p_all_px**2 + p_all_py**2 + p_all_pz**2),
            0.0
        ))[:, None]

        # Scalar sum of the pT of the selected particles in the event
        h_t = np.sum(part_pt, axis=1, keepdims=True)

        # Features of leading jet
        lead_pt = part_pt[:, 0:1]
        lead_eta = part_eta[:, 0:1]

        # Pairwise invariant mass for the two leading particles in the jet
        p12_E  = E[:, 0] + E[:, 1]
        p12_px = px[:, 0] + px[:, 1]
        p12_py = py[:, 0] + py[:, 1]
        p12_pz = pz[:, 0] + pz[:, 1]

        m_12 = np.sqrt(np.maximum(
            p12_E**2 - (p12_px**2 + p12_py**2 + p12_pz**2),
            0.0
        ))[:, None]

        # Calculate standard deviation of eta and phi of the particles in the jet
        eta_std = np.std(part_eta, axis=1, keepdims=True)
        phi_std = np.std(part_phi, axis=1, keepdims=True)

        if met:
            met_pt_feat = met_pt
            met_phi_feat = met_phi

            met_mT = np.sqrt(
                2 * jet_pt * met_pt * (1 - np.cos(jet_phi - met_phi))
            )
        else:
            met_pt_feat = np.zeros_like(h_t)
            met_phi_feat = np.zeros_like(h_t)
            met_mT = np.zeros_like(h_t)

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

        # Necessary for ml features
        X = np.stack([
            d_eta,
            d_phi,
            log_pt,
            log_E,
            log_pt_rel,
            log_E_rel,
            dR,
        ], axis=-1)

        # # Optional additional features to add to the data
        # high_level_features = np.concatenate([
        #     m_bjf,
        #     m_top,
        #     m_ll,
        #     m_jj,
        #     m_bj,
        #     m_all,
        #     eta_fj,
        #     eta_j_rad,
        #     btag_score,
        #     deltaR_fjZ,
        #     pt_Z,
        #     pt_W,
        #     pt_top,
        #     h_t
        # ], axis=-1)

        high_level_features = np.concatenate([
            m_all,
            h_t,
            lead_pt,
            lead_eta,
            m_12,
            eta_std,
            phi_std,
            met_pt_feat,
            met_phi_feat,
            met_mT
        ], axis=-1)

        # TODO: maybe try a different normalization strategy for the high level features
        # Normalize features between 0 and 1 to stabalize model training, using min-max normalization
        X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
        high_level_features = (high_level_features - np.min(high_level_features, axis=0)) / (np.max(high_level_features, axis=0) - np.min(high_level_features, axis=0) + 1e-8)

        return X, high_level_features


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


    def _convert_data(self, source: str, model: str = "ParT", met: bool=False) -> None:
        """
        Convert data containing 4 kinematics to data with 
        4 kinematics + pt eta phi + engineered features (ML based and 'normal')

        :param source: location of data folder (same as particle experiments)
        :param model: which model to use for ml features, either "MIParT" or "ParT"
        :param met: whether source data contains missing energy (if True, will add MET features to the data and pass through the model)
        """
        source = Path(source)
        
        # Load train and test data
        print(f"Loading data from {source}")
        x_train_ratio = np.load(source / "x_train_ratio.npy")
        x_train_score = np.load(source / "x_train_score.npy")
        # Three distinct test samples that all need their own preprocessed
        # features — x_test_hlvl.npy was previously only built from x_test.npy,
        # so ratio/local eval was silently misaligned.
        x_test = np.load(source / "x_test.npy")
        x_test_ratio = np.load(source / "x_test_ratio.npy")
        x_test_score = np.load(source / "x_test_score.npy")
        print(
            f"Train shapes: ratio={x_train_ratio.shape} score={x_train_score.shape}  "
            f"Test shapes: sm={x_test.shape} ratio={x_test_ratio.shape} score={x_test_score.shape}"
        )

        if not met and any(a.shape[1] % 4 != 0 for a in (x_train_ratio, x_test, x_test_ratio, x_test_score)):
            print("Possible MET features found in data but 'met' flag is not set. Please set 'met=True' to handle MET features.")
            x_train_ratio = x_train_ratio[:, :-2]
            x_train_score = x_train_score[:, :-2]
            x_test = x_test[:, :-2]
            x_test_ratio = x_test_ratio[:, :-2]
            x_test_score = x_test_score[:, :-2]

        # Transform every split
        print(f"Transforming data...")
        train_ratio_inference_data, train_ratio_features = self._transform(x_train_ratio, met=met)
        train_score_inference_data, train_score_features = self._transform(x_train_score, met=met)
        test_inference_data, test_features = self._transform(x_test, met=met)
        test_ratio_inference_data, test_ratio_features = self._transform(x_test_ratio, met=met)
        test_score_inference_data, test_score_features = self._transform(x_test_score, met=met)

        # Swap axes for model input (nsamples, n_features, max_particles)
        train_ratio_inference_data = train_ratio_inference_data.swapaxes(1, 2)
        train_score_inference_data = train_score_inference_data.swapaxes(1, 2)
        test_inference_data = test_inference_data.swapaxes(1, 2)
        test_ratio_inference_data = test_ratio_inference_data.swapaxes(1, 2)
        test_score_inference_data = test_score_inference_data.swapaxes(1, 2)

        # Initialize model
        print(f"Initializing {model} model...")
        model_instance = self._initialize_model(model)
        if model_instance is None:
            return

        # Inference jet scores for every split
        print(f"Running inference on ratio train data...")
        train_jet_scores = self._inference_jet_scores(model_instance, train_ratio_inference_data)

        print(f"Running inference on score train data...")
        train_score_jet_scores = self._inference_jet_scores(model_instance, train_score_inference_data)

        print(f"Running inference on SM test data...")
        test_jet_scores = self._inference_jet_scores(model_instance, test_inference_data)

        print(f"Running inference on ratio test data...")
        test_ratio_jet_scores = self._inference_jet_scores(model_instance, test_ratio_inference_data)

        print(f"Running inference on score test data...")
        test_score_jet_scores = self._inference_jet_scores(model_instance, test_score_inference_data)

        # Concatenate jet scores to features
        train_ratio_tf_data = np.concatenate([train_ratio_features, train_jet_scores[:, :4]], axis=-1)
        train_score_tf_data = np.concatenate([train_score_features, train_score_jet_scores[:, :4]], axis=-1)
        test_tf_data = np.concatenate([test_features, test_jet_scores[:, :4]], axis=-1)
        test_ratio_tf_data = np.concatenate([test_ratio_features, test_ratio_jet_scores[:, :4]], axis=-1)
        test_score_tf_data = np.concatenate([test_score_features, test_score_jet_scores[:, :4]], axis=-1)

        # Save transformed data (one hlvl/llvl file per raw test sample)
        print(f"Saving transformed data to {source}")
        np.save(source / "x_train_ratio_hlvl.npy", train_ratio_tf_data)
        np.save(source / "x_train_score_hlvl.npy", train_score_tf_data)
        np.save(source / "x_train_ratio_llvl.npy", train_jet_scores)
        np.save(source / "x_train_score_llvl.npy", train_score_jet_scores)
        np.save(source / "x_test_llvl.npy", test_jet_scores)
        np.save(source / "x_test_hlvl.npy", test_tf_data)
        np.save(source / "x_test_ratio_llvl.npy", test_ratio_jet_scores)
        np.save(source / "x_test_ratio_hlvl.npy", test_ratio_tf_data)
        np.save(source / "x_test_score_llvl.npy", test_score_jet_scores)
        np.save(source / "x_test_score_hlvl.npy", test_score_tf_data)
