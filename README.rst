************************************************************
Szablon kodu do uczenia sieci neuronowej w PyTorch Lightning
************************************************************

Cechy
-----

* Organizacja kodu za pomocą PyTorch Lightning.
* Wczytywanie danych za pomocą `datamodules` z PyTorch Lightning Bolts.
* Konfiguracja eksperymentów za pomocą pakietu Hydra.
* Wizualizacja metryk za pomocą loggera Weights & Biases (wandb.ai).
* Możliwość przechowywania checkpointów jako artefakty w wandb.ai.
* Formatowanie logów/konsoli za pomocą pakietu `rich`.
* Adnotacje typowania dla większości kodu.
* Zgodność z Python 3.9.


Linki
-----

* https://pytorch-lightning.readthedocs.io/en/latest/
* https://lightning-bolts.readthedocs.io/en/latest/
* https://hydra.cc/docs/next/tutorials/intro/
* https://docs.wandb.ai/
* https://github.com/willmcgugan/rich


Instalacja
----------

Po sklonowaniu repozytorium tworzenie środowiska conda::

    $ conda env create -f environment.yml
    $ conda activate zzsn

Konfiguracja środowiska w pliku `.env` po utworzeniu konta na wandb.ai::

    DATA_DIR=datasets
    RESULTS_DIR=results
    WANDB_ENTITY=WANDB_LOGIN
    WANDB_PROJECT=WANDB_PROJECT_NAME

`DATA_DIR` jest katalogiem nadrzędnym do przechowywania zbiorów danych.
`RESULTS_DIR` jest katalogiem nadrzędnym dla katalogów roboczych.


Uruchamianie eksperymentów
--------------------------

Uruchomienie uczenia z katalogu głównego (plik `.env` musi znaleźć się na ścieżce wyszukiwania)::

    $ python -m zzsn2021.main

Dodanie metadanych dla danego uruchomienia::
    
    $ python -m zzsn2021.main notes="Opis wariantu eksperymentu" tags="[TAG1, TAG2]"

Zmiana pojedynczych ustawień::

    $ python -m zzsn2021.main pl.max_epochs=150 experiment.batch_size=64

Zmiana całościowych ustawień eksperymentu na podstawie pliku YAML
(np. `src/zzsn2021/configs/experiment/fashion.yaml`)::

    $ python -m zzsn2021.main experiment=fashion

*Pliki YAML są walidowane przez schematy definiowane jako `OmegaConf structured config <https://omegaconf.readthedocs.io/en/latest/structured_config.html>`_
(np. `src/zzsn2021/configs/experiment/__init__.py`).*

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
