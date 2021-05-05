from src.data_process.load_data import DataCreator

import pandas as pd
import numpy as np
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from tsfresh.feature_extraction import extract_features


def create_train_val_test(cur_model_type, debug=False):
    """
    Подготовить x, y данные для генератора данных, где
        x - массив данных
        y - метки объектов

    :param cur_model_type: str: один из типов данных для тренировки:
                                * 'all_types' - данные и метки к ним это типы целей
                                * 'drone' - данные и метки к ним это классы внтури типа 'drone'
                                * 'fighter' - данные и метки к ним это классы внтури типа 'fighter'
                                * 'helicopter' - данные и метки к ним это классы внтури типа 'helicopter'
                                * 'missile' - данные и метки к ним это классы внтури типа 'missile'
                                * 'all_classes' - данные и метки всех классов каждого типа
    :param debug: bool: флаг для отладки

    :return: tuple: список с парами x, y для train, val, test стадий обучения соответственно
    """

    # создать или загрузить data frame
    df = DataCreator().create_dataframe()

    # добавить тип цели: ракета, самолет, вертолет, беспилотник
    mapping_type_models = {'Tomahawk_BGM': 'missile',
                           'F35A': 'fighter',
                           'F22_raptor': 'fighter',
                           'Harpoon': 'missile',
                           'AH-1Cobra': 'helicopter',
                           'Aerob': 'drone',
                           'Jassm': 'missile',
                           'EuroFighterTyphoon': 'fighter',
                           'ExocetAM39': 'missile',
                           'Dassaultrafale': 'fighter',
                           'F16': 'fighter',
                           'AH-1WSuperCobra': 'helicopter',
                           'Mig29': 'fighter',
                           'Orlan': 'drone'}

    df['Type'] = df['Class']
    df = df.replace({'Type': mapping_type_models})

    classes = np.unique(df['Type'].values)
    mapping_types = {model: idx for idx, model in enumerate(classes)}
    df['Id_Type'] = df['Type']
    df = df.replace({'Id_Type': mapping_types})

    if cur_model_type in ['all_types', 'all_classes']:
        cur_data = df  # все данные
    elif cur_model_type in ['drone', 'fighter', 'helicopter', 'missile']:
        cur_data = df[df.Model_type == cur_model_type]  # заданный тип цели
    else:
        raise ValueError('invalid value for data type')

    if debug:
        cur_data = cur_data.sample(102)

    numeric_data = np.array([np.fromstring(elem[1:-1], dtype=float, sep=',') for elem in cur_data['RCS']])
    numeric_data = numeric_data.reshape((-1, 100))

    new_df = feature_engineering(numeric_data)

    cur_data = cur_data.drop(columns=['RCS', 'Distance'])
    df = np.hstack([new_df.values, cur_data.values])
    df = pd.DataFrame(df, columns=[*new_df.columns.values, *cur_data.columns.values])

    return df


def feature_engineering(df):
    # data formation
    # id | time | value
    tmp_val = np.array([np.array([int(0)] * 100), np.array(range(100)), df[0]]).T
    for i in range(1, df.shape[0]):
        tmp_val = np.vstack([tmp_val, np.array([np.array([int(i)] * 100), np.array(range(100)), df[i]]).T])

    tmp_val = pd.DataFrame(tmp_val, columns=['id', 'time', 'value'])

    # извлечь фичи
    x = extract_features(tmp_val, default_fc_parameters=MinimalFCParameters(),
                         column_id="id", column_sort="time", column_kind=None, column_value="value",
                         n_jobs=2)

    return x