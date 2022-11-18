from datainterface.baseinterface import BaseInterface

class CurriculumConditions(BaseInterface):
    def __init__(self, env, seed, trials=10):
        features = ['x1', 'x2', 'a1', 'a2', 'b1', 'b2']
        super().__init__(env, seed, features, '/curriculumconditions')
        self.trials = trials
