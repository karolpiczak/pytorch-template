************************************************************
Szablon kodu do uczenia sieci neuronowej w PyTorch Lightning
************************************************************

Przykład uczenia prostej sieci splotowej do klasyfikacji obrazów ze zbioru `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_.

.. image:: figures/run-fashion-mnist.png
  :width: 1000
  :alt: Model training on Fashion-MNIST

Cechy
-----

* Organizacja kodu za pomocą `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_.
* Wczytywanie danych za pomocą `datamodules` z `PyTorch Lightning Bolts <https://lightning-bolts.readthedocs.io/en/latest/>`_.
* Konfiguracja eksperymentów za pomocą pakietu `Hydra <https://hydra.cc/docs/next/tutorials/intro/>`_.
* Wizualizacja metryk za pomocą `Weights & Biases (wandb.ai) <https://docs.wandb.ai/>`_.
* Możliwość przechowywania checkpointów jako artefakty w wandb.ai.
* Formatowanie logów/informacji w konsoli za pomocą pakietu `rich <https://github.com/willmcgugan/rich>`_.
* Adnotacje typowania dla większości kodu.
* Zgodność z Python 3.9.

Instalacja
----------

Po sklonowaniu repozytorium tworzenie środowiska conda::

    $ conda env create -f environment.yml
    $ conda activate zzsn

Konfiguracja środowiska w pliku ``.env`` po utworzeniu konta na wandb.ai::

    DATA_DIR=datasets
    RESULTS_DIR=results
    WANDB_ENTITY=WANDB_LOGIN
    WANDB_PROJECT=WANDB_PROJECT_NAME

``DATA_DIR`` jest katalogiem nadrzędnym do przechowywania zbiorów danych.
``RESULTS_DIR`` jest katalogiem nadrzędnym dla katalogów roboczych.


Uruchamianie eksperymentów
--------------------------

Uruchomienie uczenia z katalogu głównego (plik ``.env`` musi znaleźć się na ścieżce wyszukiwania)::

    $ python -m zzsn2021.main

Dodanie metadanych dla danego uruchomienia::
    
    $ python -m zzsn2021.main notes="Opis wariantu eksperymentu" tags="[TAG1, TAG2]"

Zmiana pojedynczych ustawień::

    $ python -m zzsn2021.main pl.max_epochs=150 experiment.batch_size=64

Zmiana całościowych ustawień eksperymentu na podstawie pliku YAML
(np. ``src/zzsn2021/configs/experiment/fashion.yaml``)::

    $ python -m zzsn2021.main experiment=fashion

Pliki YAML są walidowane przez schematy definiowane jako `OmegaConf structured config <https://omegaconf.readthedocs.io/en/latest/structured_config.html>`_
(np. ``src/zzsn2021/configs/experiment/__init__.py``).

Wyłączenie przesyłania danych do *wandb*::

    $ WANDB_MODE=dryrun python -m zzsn2021.main 

Tryb *debug* (całkowicie wyłącza logowanie *wandb*)::

    $ RUN_MODE=debug python -m zzsn2021.main


Upload checkpointów do *wandb*
------------------------------

`wandb.ai <https://wandb.ai>`_ pozwala na zdalne przechowywanie i współdzielenie plików (np. wyuczonych modeli)
za pomocą funkcjonalności `W&B Artifacts <https://docs.wandb.ai/guides/artifacts/api>`_.

Do przesyłania checkpointów wygenerowanych w czasie uczenia można wykorzystać skrypt `cli.py`::

    $ python -m zzsn2021.cli upload RUN_NAME CHECKPOINT_NAME ARTIFACT_NAME

Przykładowo::

    $ python -m zzsn2021.cli upload 20210422-1430-phenomenal-elephant epoch_5.ckpt test_model

Tak przesłane pliki można później wykorzystać do wznowienia uczenia na innym systemie przez zmianę
odpowiedniego ustawienia w konfiguracji eksperymentu, przykładowo::

    resume_checkpoint: wandb://WANDB_USER/WANDB_PROJECT/test_model:v0@epoch_5.ckpt
