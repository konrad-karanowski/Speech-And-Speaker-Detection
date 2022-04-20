from dataset import SpeechDataModule
from models import SiameseModel
from pytorch_lightning.loggers import WandbLogger

from utils import train_test


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
    'root': r'C:\Users\Konrad\Desktop\PythonProjects\Speech-And-Speaker-Detection\data',
    'data_dir': 'archive',
    'img_dir': 'mel_spectrogram',

    'frame_size': 2048,
    'hop_size': 128,
    'window_function': 'hann',

    'batch_size': 16,

    'target_sr': 22050,
    'target_num_samples': 22050,
    'res_type': 'kaiser_best',

    'input_shape': (128, 128)
})


def main():
    datamodule = SpeechDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()

    model = SiameseModel(config)

    logger = WandbLogger(
        config=config,
        entity='kn-bmi',
        project='TEST',
        log_model=False
    )

    train_test(model, datamodule, logger, config)


if __name__ == '__main__':
    main()
