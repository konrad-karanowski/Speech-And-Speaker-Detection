from typing import List
import os
import glob

import pandas as pd


def get_classes(path: str) -> List[str]:
    """Read classes list using glob

    Args:
        path (str): path to root data directory

    Returns:
        List[str]: list of classes
    """
    return [k.split('/')[-1] for k, _, _ in os.walk(path) if not '_background_noise_' in k and 'archive/' in k]


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
            splitted = file.split('/')[-1].split('_')
            record = file, label, speaker, nohash, utterance_id = file, cls, splitted[0], splitted[1], splitted[2][:-4]
            data.append(record)
    df = pd.DataFrame(data=data, columns=['path', 'label', 'speaker_id', 'hash', 'utterance_id'])
    return df

