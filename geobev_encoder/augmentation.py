"""Scene-consistent photometric augmentation, isolated from model RNG streams."""
import copy
import math

import torch


def color_config(probability=0., strength=.1):
    probability, strength = float(probability), float(strength)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Color probability must be in [0,1]')
    if not math.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('Color strength must be in [0,1)')
    if probability == 0:
        return None
    return dict(schema='scene_rgb_color_v1', probability=probability, strength=strength,
                order=['brightness', 'contrast', 'saturation'],
                shared='all_cameras_and_frames_in_continuous_scene_segment',
                world_augmented=False, evaluation_enabled=False)


def canonical_color_config(config):
    if config is None:
        return None
    expected = color_config(config.get('probability', -1), config.get('strength', -1))
    if config != expected:
        raise ValueError('Unsupported color augmentation contract')
    return expected


def apply_color(images, factors):
    """Input/output follow the pretrained GeoBEV BGR/mean/std convention."""
    if factors is None or all(value == 1. for value in factors):
        return images
    with torch.autocast(device_type=images.device.type, enabled=False):
        mean = images.new_tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
        std = images.new_tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
        rgb = ((images.float() * std + mean).flip(-3) / 255.).clamp(0, 1)
        brightness, contrast, saturation = factors
        rgb = (rgb * brightness).clamp(0, 1)
        gray_weights = rgb.new_tensor([.2989, .587, .114])[:, None, None]
        gray_mean = (rgb * gray_weights).sum(-3, keepdim=True).mean((-2, -1), keepdim=True)
        rgb = (rgb * contrast + gray_mean * (1 - contrast)).clamp(0, 1)
        gray = (rgb * gray_weights).sum(-3, keepdim=True)
        rgb = (rgb * saturation + gray * (1 - saturation)).clamp(0, 1)
        return ((rgb.flip(-3) * 255. - mean) / std).to(images.dtype)


class SceneColorAugmentation:
    def __init__(self, config, seed):
        self.config = canonical_color_config(config)
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.slots = []

    def reset(self):
        self.slots = []

    def prepare(self, metas):
        if len(self.slots) != len(metas):
            self.slots = [None for _ in metas]
        factors = []
        for index, meta in enumerate(metas):
            previous = self.slots[index]
            continuous = previous is not None and previous['scene_token'] == meta['scene_token'] and abs(
                float(meta['timestamp']) - previous['timestamp'] - .5) <= .1
            if not continuous:
                draws = torch.rand(4, generator=self.generator).tolist()
                enabled = draws[0] < self.config['probability']
                strength = self.config['strength']
                value = tuple(1 + (2 * d - 1) * strength for d in draws[1:]) if enabled else None
                previous = dict(scene_token=meta['scene_token'], timestamp=float(meta['timestamp']),
                                factors=value)
            previous['timestamp'] = float(meta['timestamp'])
            self.slots[index] = previous
            factors.append(previous['factors'])
        return factors

    def state_dict(self):
        return dict(config=self.config, generator=self.generator.get_state().clone(),
                    slots=copy.deepcopy(self.slots))

    def load_state_dict(self, state):
        if state['config'] != self.config:
            raise ValueError('Color augmentation resume configuration differs')
        self.generator.set_state(state['generator'].cpu())
        self.slots = copy.deepcopy(state['slots'])
