"""Infinite ordered scene streams, with a consumed cursor independent of prefetch."""
import copy
import random


class SceneStreamBatchSampler:
    def __init__(self, dataset, batch_size, rank=0, world_size=1, seed=2026):
        self.scenes = list(dataset.scenes.values())
        self.batch_size, self.rank, self.world_size, self.seed = batch_size, rank, world_size, seed
        if len(self.scenes) < batch_size * world_size:
            raise ValueError('need at least one scene per global stream')
        self.consumed = 0
        self.committed = None
        self._pending = {}

    def _initial(self):
        order = list(range(len(self.scenes)))
        random.Random(self.seed).shuffle(order)
        total = self.batch_size * self.world_size
        result = []
        for slot in range(self.batch_size):
            global_slot = self.rank * self.batch_size + slot
            result.append({'pool': order[global_slot::total], 'scene_pos': 0, 'frame': 0,
                           'cycle': 0, 'slot': global_slot})
        return result

    def __iter__(self):
        streams = copy.deepcopy(self.committed if self.committed is not None else self._initial())
        cursor = self.consumed
        while True:
            batch = []
            for stream in streams:
                if stream['scene_pos'] == len(stream['pool']):
                    stream['cycle'] += 1
                    random.Random(self.seed + 100003 * stream['cycle'] + stream['slot']).shuffle(stream['pool'])
                    stream['scene_pos'] = stream['frame'] = 0
                scene = self.scenes[stream['pool'][stream['scene_pos']]]
                batch.append(scene[stream['frame']])
                stream['frame'] += 1
                if stream['frame'] == len(scene):
                    stream['scene_pos'] += 1
                    stream['frame'] = 0
            cursor += 1
            self._pending[cursor] = copy.deepcopy(streams)
            yield batch

    def commit(self):
        self.consumed += 1
        self.committed = self._pending.pop(self.consumed)

    def state_dict(self):
        return {'consumed': self.consumed, 'streams': self.committed,
                'identity': (self.batch_size, self.rank, self.world_size, self.seed)}

    def load_state_dict(self, state):
        if tuple(state['identity']) != (self.batch_size, self.rank, self.world_size, self.seed):
            raise ValueError('exact stream resume requires same batch/rank/world/seed')
        self.consumed, self.committed = state['consumed'], copy.deepcopy(state['streams'])
        self._pending = {}
