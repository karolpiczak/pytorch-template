************************************************************
PyTorch Lightning neural network code template
************************************************************

An example of training a simple convolutional neural network for image classification on the `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ dataset.

This code template, showcasing my preferred coding style for AI/ML experiments, was crafted for my students enrolled in the `Advanced Topics in Neural Networks - 2021 <https://pw.karolpiczak.com/dydaktyka/2021l-zzsn-projekty>`_ course.

.. image:: figures/run-fashion-mnist.png
  :width: 1000
  :alt: Model training on Fashion-MNIST

Features
-------

* Code organization using `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_.
* Data loading via `datamodules` from `PyTorch Lightning Bolts <https://lightning-bolts.readthedocs.io/en/latest/>`_.
* Experiment configuration with the `Hydra <https://hydra.cc/docs/next/tutorials/intro/>`_ package.
* Metric visualization using `Weights & Biases (wandb.ai) <https://docs.wandb.ai/>`_.
* Ability to store checkpoints as artifacts in wandb.ai.
* Console logs/information formatting using the `rich <https://github.com/willmcgugan/rich>`_ package.
* Type annotations for most of the code.
* Compatible with Python 3.9.

Installation
----------

After cloning the repository, set up the conda environment::

    $ conda env create -f environment.yml
    $ conda activate zzsn

Configure the environment in the ``.env`` file after creating an account on wandb.ai::

    DATA_DIR=datasets
    RESULTS_DIR=results
    WANDB_ENTITY=WANDB_LOGIN
    WANDB_PROJECT=WANDB_PROJECT_NAME

``DATA_DIR`` is the parent directory for storing datasets.
``RESULTS_DIR`` is the parent directory for workspace directories.

Running Experiments
--------------------------

To start training from the main directory (the ``.env`` file should be in the search path)::

    $ python -m zzsn2021.main

To add metadata for a specific run::

    $ python -m zzsn2021.main notes="Experiment variant description" tags="[TAG1, TAG2]"

To change individual settings::

    $ python -m zzsn2021.main pl.max_epochs=150 experiment.batch_size=64

To change overall experiment settings based on a YAML file
(e.g., ``src/zzsn2021/configs/experiment/fashion.yaml``)::

    $ python -m zzsn2021.main experiment=fashion

YAML files are validated by schemas defined as `OmegaConf structured config <https://omegaconf.readthedocs.io/en/latest/structured_config.html>`_
(e.g., ``src/zzsn2021/configs/experiment/__init__.py``).

To disable data transfer to *wandb*::

    $ WANDB_MODE=dryrun python -m zzsn2021.main 

*Debug* mode (completely disables *wandb* logging)::

    $ RUN_MODE=debug python -m zzsn2021.main

Uploading Checkpoints to *wandb*
------------------------------

`wandb.ai <https://wandb.ai>`_ allows for remote storage and sharing of files (e.g., trained models)
using the `W&B Artifacts <https://docs.wandb.ai/guides/artifacts/api>`_ functionality.

To upload checkpoints generated during training, you can use the `cli.py` script::

    $ python -m zzsn2021.cli upload RUN_NAME CHECKPOINT_NAME ARTIFACT_NAME

For example::

    $ python -m zzsn2021.cli upload 20210422-1430-phenomenal-elephant epoch_5.ckpt test_model

Files uploaded this way can later be used to resume training on another system by changing
the appropriate setting in the experiment configuration, for example::

    resume_checkpoint: wandb://WANDB_USER/WANDB_PROJECT/test_model:v0@epoch_5.ckpt
