import math
import numpy as np
import pandas as pd

def Dataset():
    data = pd.read_csv('./SEER_DATA.csv', sep=",")
    data['Vital status'] = np.where(data['Vital status'] == 'Dead', 1, 0)
    data['Sex'] = np.where(data['Sex'] == 'Male', 1, 0)
    data['Marital status at diagnosis'] = np.where(data['Marital status at diagnosis'] == 'Married (including common law)', 1, 0)
    data['Primary'] = np.where(data['Primary'] == 'Yes', 1, 0)
    data['Separate Tumor Nodules Ipsilateral Lung'] = np.where(data['Separate Tumor Nodules Ipsilateral Lung'] == 'None; No intrapulmonary mets; Foci in situ/minimally invasive adenocarcinoma', 0, 1)
    data['Chemotherapy'] = np.where(data['Chemotherapy'] == 'Yes', 1, 0)
    data['Age'] = data['Age'].str.replace(r'\s*years', '', regex = True).astype(np.float64)

    data = np.array(data, dtype = np.float64)
    rng = np.random.RandomState(3407)
    rng.shuffle(data)

    sample_size = data.shape[0]
    split_index = math.ceil(0.8 * sample_size)

    train_time = data[: split_index, 0] / 12
    test_time = data[split_index: , 0] / 12

    train_event = data[: split_index, 1]
    test_event = data[split_index: , 1]

    train_z = data[: split_index, 2: 7]
    test_z = data[split_index: , 2: 7]

    train_x = data[: split_index, 7: ]
    test_x = data[split_index: , 7: ]
    train_x = (train_x - train_x.min(0)) / (train_x.max(0) - train_x.min(0))
    test_x = (test_x - test_x.min(0)) / (test_x.max(0) - test_x.min(0))

    return train_z, test_z, train_x, test_x, train_time, test_time, train_event, test_event