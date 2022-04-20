from typing import List
import os
import sys
import glob

import pandas as pd


def get_classes(path: str, ignored: List[str] = ['_background_noise_']) -> List[str]:
    """Read classes list using glob

    Args:
        path (str): path to root data directory
        ignored (List[str]): list of strings that cannot be involved in path 

    Returns:
        List[str]: list of classes
    """
    return [os.path.split(k.path)[-1] for k in os.scandir(path) if not any(x in k.path for x in ignored)]


def create_df(classes: List[str], path: str) -> pd.DataFrame:
    """Creates dataframe with description and info 

    Args:
        classes (List[str]): list of classes
        path (str): path to root directory

    Returns:
        pd.DataFrame: dataframe containing [path to file, label, speaker, hash, utterance_id]
    """
    data = []
    for cls in classes:
        files = glob.glob(os.path.join(path, cls, '*'))
        for file in files:
            splitted = os.path.split(file)[-1].split('_')
            record = file, label, speaker, nohash, utterance_id = file, cls, splitted[0], splitted[1], splitted[2][:-4]
            data.append(record)
    df = pd.DataFrame(data=data, columns=['path', 'label', 'speaker_id', 'hash', 'utterance_id'])
    return df

