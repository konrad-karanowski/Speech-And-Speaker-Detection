from dataset import SpeechDataModule
import matplotlib.pyplot as plt

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
    'window_function': 'hann',

    'batch_size': 32,

    'target_sr': 22050,
    'target_num_samples': 22050,
    'res_type': 'kaiser_best',
})


def main():
    datamodule = SpeechDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()

    train = datamodule.train_dataloader()

    i = 0
    for l in train:
        x, y, z = l['anchor'][0, :], l['positive'][0, :], l['negative'][0, :]
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title(l['positive_label'][0])
        ax[0].imshow(x.squeeze(0))
        ax[1].set_title(l['positive_label'][0])
        ax[1].imshow(y.squeeze(0))
        ax[2].set_title(l['negative_label'][0])
        ax[2].imshow(z.squeeze(0))
        plt.show()
        break


if __name__ == '__main__':
    main()
