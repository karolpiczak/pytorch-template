from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
import wandb
from click import ClickException
from dotenv import load_dotenv
from rich import print

load_dotenv()


@click.group()
def main():
    pass


@main.command()  # type: ignore
@click.argument('run_name')
@click.argument('checkpoint_name')
@click.argument('artifact_name')
def upload(run_name: str, checkpoint_name: str, artifact_name: str) -> None:
    """
    Upload model checkpoint (CHECKPOINT_NAME) from a given run (RUN_NAME)
    as W&B artifact (ARTIFACT_NAME).
    """

    run_dir = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}/{run_name}'

    print(f'Loading run data from: [bold yellow]{run_dir}')

    wandb_dir = Path(f'{run_dir}/wandb/latest-run/')
    if not wandb_dir.exists():
        raise ClickException(f'Could not find {wandb_dir}.')

    glob = list(wandb_dir.glob('*.wandb'))
    if not len(glob):
        raise ClickException(f'Could not find a unique *.wandb file.')

    wandb_id = glob[0].with_suffix('').name.split('-')[1]
    print(f'Run ID: [bold yellow]{wandb_id}')

    checkpoint_file = Path(f'{run_dir}/checkpoints/{checkpoint_name}')
    if not checkpoint_file.exists():
        raise ClickException(f'Could not find {checkpoint_file}.')

    run: Any = wandb.init(project=os.getenv('WANDB_PROJECT'), entity=os.getenv('WANDB_ENTITY'),  # type: ignore
                          resume='must', id=wandb_id)

    artifact: Any = wandb.Artifact(f'{artifact_name}', type='checkpoint')
    artifact.add_file(str(checkpoint_file))

    run.log_artifact(artifact)


if __name__ == '__main__':
    main()
