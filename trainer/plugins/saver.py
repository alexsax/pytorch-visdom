from .plugin import Plugin


class Saver(Plugin):
    stat_name = 'progress'

    def __init__(self, filename, interval=(1, 'epoch'), should_save=lambda x: True):
        super(Saver, self).__init__([interval])
        self.filename = filename
        self.should_save = should_save

    def register(self, trainer):
        self.trainer = trainer
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['samples_used'] = 0
        stats['epoch_size'] = len(trainer.dataset)
        stats['log_iter_fields'] = [
            '{samples_used}/{epoch_size}',
            '({percent:.2f}%)'
        ]

    def iteration(self, iteration, input, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['samples_used'] += 1
        stats['percent'] = 100. * stats['samples_used'] / stats['epoch_size']

    def epoch(self, *args):
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['samples_used'] = 0
        stats['percent'] = 0
