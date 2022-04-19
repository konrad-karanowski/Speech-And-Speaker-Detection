from dataset import SpeechDataModule


class AttributeDict(dict):
    """
    Class allowing access dictionary items as properties. 
    For example:
    
    my_dict = AttributeDict({'name': 'Adam', 'salary': 30})
    print(my_dict.name)
    >> Adam
    print(my_dict.salary)
    >> 30
    my_dict.profession = 'Worker'
    print(my_dict.profession)
    >> Worker
    """
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

config = AttributeDict({
    'root': 'data',
    'data_dir': 'archive',
    'img_dir': 'mel_spectrogram',

    'frame_size': 2048,
    'hop_size': 128,
    'window_function': 'hann'
})


def main():
    datamodule = SpeechDataModule(config)
    datamodule.prepare_data()


if __name__ == '__main__':
    main()
