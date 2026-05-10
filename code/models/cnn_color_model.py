"""Phase 1 model placeholder — network implemented in Task-05."""
from .base_model import BaseModel


class CnnColorModel(BaseModel):
    def name(self):
        return 'CnnColorModel'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        super().initialize(opt)
        self.model_names = []   # filled in Task-05

    def set_input(self, data):
        pass

    def forward(self):
        pass

    def optimize_parameters(self):
        pass
