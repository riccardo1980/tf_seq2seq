import tensorflow as tf
import os

_datasets = {
    'anki-spa-eng': {
        'url':
            'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        'text_file': 'spa-eng/spa.txt'
    }
}


def dataset_download(dataset_name: str) -> str:

    # Download the file
    if dataset_name in _datasets:
        dataset_descr = _datasets[dataset_name]

    path_to_zip = tf.keras.utils.get_file(
        os.path.split(dataset_descr['url'])[-1],
        origin=dataset_descr['url'],
        extract=True
    )

    path_to_file = os.path.join(os.path.dirname(path_to_zip),
                                dataset_descr['text_file'])

    return path_to_file
