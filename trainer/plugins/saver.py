from .plugin import Plugin
import torch

class Saver(Plugin):
    stat_name = 'progress'

    def __init__(self, filename, interval=(1, 'epoch'), should_save=lambda x: True):
        ''' 
            Args:
                filename: The filename to save the model under. This can contain format 
                    variables, and filename.format(**save_dict) will be called.
                should_save: A function which takes in 'save_dict' and returns either False,
                    if it should not save, or the either True (in which case it will save 
                    under 'filename'), or alternatively the filename can be returned.

                    should_save will have access to the 'trainer', which also gives it access
                    to all of the saved stats, which can be used to make the decision whether 
                    to save.
        '''
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

    def make_param_dict(self):
        param_dict = {
            'state_dict': self.trainer.model.state_dict(),
            'optimizer' : self.trainer.optimizer.state_dict(),
            'saved_stats': self.trainer.statse
        }
        if not param_dict:
            param_dict = {
                self.trainer.model
            }
        torch.save(param_dict, self.filename)

    def iteration(self, iteration, input, *args):

        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['samples_used'] += 1
        stats['percent'] = 100. * stats['samples_used'] / stats['epoch_size']

    def epoch(self, epoch_num):

        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['samples_used'] = 0
        stats['percent'] = 0
