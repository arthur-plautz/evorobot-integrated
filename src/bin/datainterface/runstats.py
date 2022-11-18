import numpy as np
import pandas as pd
from datainterface.baseinterface import BaseInterface

class RunStats(BaseInterface):
    def __init__(self, env, seed, features):
        features = features if features else [
            'msteps',
            'bestfit',
            'bestgfit',
            'bestsam',
            'avgfit',
            'paramsize'
        ]
        super().__init__(env, seed, features, '/runstats')
        self.metrics = []

    def __metric_format(self, metric):
        return f'{self.interface_dir}/s{self.seed}_{metric}.npy'

    @property
    def __metrics_file(self):
        return f'{self.interface_dir}/s{self.seed}_metrics.csv'

    @property
    def __test_file(self):
        return f'{self.interface_dir}/s{self.seed}_test.csv'

    def save_test(self, score, steps):
        df = pd.DataFrame([{'score': score, 'steps': steps}])
        df.to_csv(self.__test_file)

    def save_metric(self, data, metric):
        metric_file = self.__metric_format(metric)
        data = np.asarray(data)
        np.save(metric_file, data)
        if metric not in self.metrics:
            self.metrics.append(metric)

    def save(self):
        super().save()
        df = pd.DataFrame()
        for metric in self.metrics:
            metric_file = self.__metric_format(metric)
            m_df = np.load(metric_file)
            df[metric] = m_df
        if not df.empty:
            df.to_csv(self.__metrics_file)
