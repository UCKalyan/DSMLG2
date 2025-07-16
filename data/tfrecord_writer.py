import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.helpers import load_npy, ensure_dir

logger = get_logger(__name__)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(volume, segmentation, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {
        'volume': _bytes_feature(tf.io.serialize_tensor(volume)),
        'segmentation': _bytes_feature(tf.io.serialize_tensor(segmentation)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
        'height': _int64_feature(volume.shape[0]),
        'width': _int64_feature(volume.shape[1]),
        'depth': _int64_feature(volume.shape[2]),
        'channels': _int64_feature(volume.shape[3]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

class TFRecordWriter:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['processed_data_path']
        self.tfrecord_path = os.path.join(self.processed_path, 'tfrecords')
        ensure_dir(self.tfrecord_path)
        self.split_ratios = config['train_val_test_split']

    def convert(self):
        """
        Converts all .npy patient data into train, validation, and test TFRecord files.
        """
        # Filter out the 'tfrecords' directory itself to avoid processing it as a patient
        patient_ids = [
            p for p in os.listdir(self.processed_path)
            if os.path.isdir(os.path.join(self.processed_path, p)) and p != 'tfrecords'
        ]
        if not patient_ids:
            logger.error("No processed patient data found to convert.")
            return

        # Split patient IDs
        train_val_ids, test_ids = train_test_split(patient_ids, test_size=self.split_ratios[2], random_state=42)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=self.split_ratios[1] / (1 - self.split_ratios[2]), random_state=42)

        logger.info(f"Splitting data: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test.")

        # Write each set to a different file
        self._write_tfrecord_file(train_ids, 'train.tfrecord')
        self._write_tfrecord_file(val_ids, 'validation.tfrecord')
        self._write_tfrecord_file(test_ids, 'test.tfrecord')

    def _write_tfrecord_file(self, patient_ids, filename):
        """Writes a set of patients to a single TFRecord file."""
        filepath = os.path.join(self.tfrecord_path, filename)
        logger.info(f"Writing to TFRecord file: {filepath}")

        with tf.io.TFRecordWriter(filepath) as writer:
            for patient_id in tqdm(patient_ids, desc=f"Converting to {filename}"):
                try:
                    patient_dir = os.path.join(self.processed_path, patient_id)
                    volume = load_npy(os.path.join(patient_dir, 'volume.npy'))
                    segmentation = load_npy(os.path.join(patient_dir, 'segmentation.npy'))
                    label = load_npy(os.path.join(patient_dir, 'label.npy'))

                    serialized_example = serialize_example(volume, segmentation, label)
                    writer.write(serialized_example)
                except Exception as e:
                    logger.error(f"Could not process patient {patient_id} for TFRecord: {e}")
