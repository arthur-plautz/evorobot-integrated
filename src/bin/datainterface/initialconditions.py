from datainterface.baseinterface import BaseInterface

class InitialConditions(BaseInterface):
    def __init__(self, env, seed, features, trials=10):
        super().__init__(env, seed, features, '/initialconditions')
        self.trials = trials
