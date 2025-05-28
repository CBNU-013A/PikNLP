from model.src import Trainer, Evaluator, Tester
import os
import json
from attrdict import AttrDict

class ModelRunner:
    def __init__(self, cli_args):
        self.args: AttrDict = self._load_config(cli_args)

    @staticmethod
    def _load_config(cli_args):
        with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
            return AttrDict(json.load(f))
        
    @classmethod
    def from_cli_args(cls, cli_args):
        runner = cls(cli_args)
        if cli_args.mode == 'train':
            trainer = Trainer(runner.args)
            trainer.train()
        elif cli_args.mode == 'eval':
            evaluator = Evaluator(runner.args)
            evaluator.evaluate()
        elif cli_args.mode == 'test':
            tester = Tester(runner.args)
            tester.test()
        elif cli_args.mode == 'predict':
            tester = Tester(runner.args)
            tester.predict()
        else:
            raise ValueError(f'Invalid mode: {cli_args.mode}')