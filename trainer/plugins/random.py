import numpy as np
from .plugin import Plugin


class RandomMonitor(Plugin):
    stat_name = 'random'

    def __init__(self, n):
        super(RandomMonitor, self).__init__([(1, 'iteration'), (1, 'epoch')])
        self.n = n

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['data'] = np.zeros(self.n)
        stats['n'] = self.n
        stats['log_iter_fields'] = [
            'mean: {mean:.2f}',
            'n: {n}'
        ]

    def iteration(self, iteration, input, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['data'] += np.random.rand(self.n)

    def epoch(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        # stats['data'] = np.zeros(self.n)
