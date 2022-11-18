from datainterface.baseinterface import BaseInterface

class BaseConditions(BaseInterface):
    def __init__(self, env, seed, total_conditions, save_dir='/baseconditions'):
        features = [i for i in range(total_conditions)]
        super().__init__(env, seed, features, save_dir)
        self.trials = 1
