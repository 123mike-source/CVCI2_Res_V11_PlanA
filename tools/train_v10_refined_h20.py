"""Reviewed V10 recipe: original pretraining, partial IRWM, t99 and refined loss."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    path = ROOT / 'configs/v10_refined/train.json'
    config = json.loads(path.read_text())
    os.environ['CVCI_TRAINING_RECIPE_FILE'] = str(path)
    defaults = config['arguments']
    # Saved trainer config is authoritative for exact resume; explicit overrides
    # are still checked against the checkpoint by the shared trainer.
    resume = None
    if '--resume' in sys.argv:
        import torch
        resume = torch.load(sys.argv[sys.argv.index('--resume') + 1], map_location='cpu')['metadata']['config']
    for key, value in defaults.items():
        option = '--' + key.replace('_', '-')
        if option in sys.argv:
            continue
        if resume is not None:
            if key not in resume:
                raise ValueError('Checkpoint predates the refined recipe: ' + key)
            value = resume[key]
        if isinstance(value, bool):
            if value:
                sys.argv.append(option)
        elif isinstance(value, list):
            sys.argv.extend([option] + [str(item) for item in value])
        elif value is not None:
            sys.argv.extend([option, str(value)])
    if '--warm-start' in sys.argv or '--warm-start-geobev' in sys.argv:
        raise ValueError('The refined recipe starts from original pretraining; use --resume only for its own last.pth')
    from tools.train_geobev_h20 import main as train
    train()


if __name__ == '__main__':
    main()
