import numpy as np
import pandas as pd
from utils import get_root_dir, create_dir, remove_file, verify_file, create_dirs

class BaseInterface:
    def __init__(self, env, seed, columns, interface_dir):
        self.columns = columns
        self.seed = seed
        self.env = env
        self.data_dir = f'{get_root_dir()}/data/'
        self.env_dir = self.data_dir + self.env
        create_dir(self.env_dir)
        self.interface_dir = self.env_dir + interface_dir
        create_dirs(self.env_dir, interface_dir)
        self.stage_dir = self.interface_dir + '/stg/'
        create_dir(self.stage_dir)
        self.stages = []

    @property
    def __empty_matrix(self):
        return [np.arange(self.__n_columns)]

    @property
    def __n_columns(self):
        return len(self.columns)

    def __stg_format(self, stage):
        return f'{self.stage_dir}/s{self.seed}_g{stage}_stg.npy'

    def __purge_stg(self):
        try:
            for stg in self.stages:
                stg_file = self.__stg_format(stg)
                remove_file(stg_file)
        except Exception as e:
            print('Error Purging staging files')

    def __stg_col(self, stg_len):
        col = []
        for i in range(len(stg_len)):
            col.append([self.stages[i]] * stg_len[i])
        if col:
            return np.concatenate(col)
        return []

    def save_stg(self, data, stage):
        stg_file = self.__stg_format(stage)
        data = data if np.array(data).ndim > 1 else [data]
        data = np.asarray(data)
        np.save(stg_file, data)
        self.stages.append(stage)

    def save(self):
        data_file = f'{self.interface_dir}/s{self.seed}_run.csv'
        save_data = self.__empty_matrix
        stg_len = []
        for stg in self.stages:
            stg_file = self.__stg_format(stg)
            if verify_file(stg_file):
                data = np.load(stg_file, allow_pickle=True)
                save_data = np.append(save_data, data, axis=0)
                stg_len.append(len(data))
        df = pd.DataFrame(save_data[1:], columns=self.columns)
        df['gen'] = self.__stg_col(stg_len)
        df.to_csv(data_file, index=False)
        self.__purge_stg()
