"""HTCondor launcher"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import htcondor2
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _tarball(cfg: DictConfig) -> Path:
    """
    Creates a tarball containing the datasets to be used in the experiment,
    the configuration yaml file and, optionally, experiment checkpoints

    :param cfg: Main configuration object
    :type cfg: DictConfig
    :return: Description
    :rtype: Path
    """
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True, parents=True)

    # Create unique safe name for tarball
    with tempfile.NamedTemporaryFile(mode="r+", dir=input_dir, suffix=".tar.gz") as tmp:
        tarball = Path(tmp.name)

    with tarfile.open(tarball, "w:gz") as tar:
        # Add data and possibly model checkpoints
        tar.add(cfg.dataset.path)
        if Path(cfg.data.run_dir).exists():
            tar.add(cfg.data.run_dir)

        # Add configuration object
        cfg_encoded = OmegaConf.to_yaml(cfg).encode()
        info = tarfile.TarInfo(name="input.yaml")
        info.size = len(cfg_encoded)
        tar.addfile(info, io.BytesIO(cfg_encoded))

    return tarball


def launch(
    *,
    description: DictConfig,
    description_addition: DictConfig = {},
    cfg: DictConfig,
    **_,
):
    """
    Launch a condor job with parameters specified in cfg and the tarball with the
    needed files.

    :param description: Main job description
    :type description: DictConfig
    :param description_addition: Additional description for our cluster.
    :type description_addition: DictConfig
    :param cfg: Description
    :type cfg: DictConfig
    """
    # Prepare job description
    job_description = htcondor2.Submit(dict(description))

    # Because of the special symbols `+` we need to update the already created
    # Submit object
    job_description.update(description_addition)

    # Create tarball with data files and configs
    tb = _tarball(cfg)
    itemdata = [{"tarball": str(tb), "tarball_name": tb.name}]

    # Schedule job
    schedd = htcondor2.Schedd()
    schedd.submit(description=job_description, itemdata=iter(itemdata))
