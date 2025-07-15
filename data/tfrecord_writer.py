import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

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

def serialize_example(volume, segmentation, label, config):
    """
    Creates a tf.train.Example message ready to be written to a file.
    Uses the volume_shape from the config to write metadata.
    """
    # --- MODIFICATION START ---
    # Get shape metadata directly from the config for consistency.
    shape = config['volume_shape']
    feature = {
        'volume': _bytes_feature(tf.io.serialize_tensor(volume)),
        'segmentation': _bytes_feature(tf.io.serialize_tensor(segmentation)),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
        'height': _int64_feature(shape[0]),
        'width': _int64_feature(shape[1]),
        'depth': _int64_feature(shape[2]),
        'channels': _int64_feature(shape[3]),
    }
    # --- MODIFICATION END ---
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

class TFRecordWriter:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['processed_data_path']
        self.tfrecord_path = os.path.join(self.processed_path, 'tfrecords')
        ensure_dir(self.tfrecord_path)

    def convert(self):
        """
        Converts all .npy patient data into a single train.tfrecord and val.tfrecord file.
        """
        patient_ids = [p for p in os.listdir(self.processed_path) if os.path.isdir(os.path.join(self.processed_path, p))]
        
        train_ids = patient_ids[:int(len(patient_ids) * 0.8)]
        val_ids = patient_ids[int(len(patient_ids) * 0.8):]

        self._write_tfrecord_file(train_ids, 'train.tfrecord')
        self._write_tfrecord_file(val_ids, 'validation.tfrecord')

    def _write_tfrecord_file(self, patient_ids, filename):
        """Writes a set of patients to a single TFRecord file."""
        filepath = os.path.join(self.tfrecord_path, filename)
        logger.info(f"Writing to TFRecord file: {filepath}")
        
        precision_str = self.config.get('precision', 'float32')
        float_dtype = np.float64 if precision_str == 'float64' else np.float32
        logger.info(f"Using {precision_str} for TFRecord conversion.")
        
        # --- MODIFICATION START ---
        # Get the expected shape from the config to use for validation.
        expected_shape = tuple(self.config['volume_shape'])
        # --- MODIFICATION END ---

        with tf.io.TFRecordWriter(filepath) as writer:
            for patient_id in tqdm(patient_ids, desc=f"Converting to {filename}"):
                try:
                    patient_dir = os.path.join(self.processed_path, patient_id)
                    volume = load_npy(os.path.join(patient_dir, 'volume.npy'))
                    segmentation = load_npy(os.path.join(patient_dir, 'segmentation.npy'))
                    label = load_npy(os.path.join(patient_dir, 'label.npy'))

                    # --- MODIFICATION START ---
                    # Validate the shape of the loaded volume against the config.
                    if volume.shape != expected_shape:
                        logger.warning(f"Skipping patient {patient_id}. Volume shape {volume.shape} does not match expected shape {expected_shape} from config.")
                        continue  # Skip to the next patient
                    # --- MODIFICATION END ---

                    volume = volume.astype(float_dtype)
                    segmentation = segmentation.astype(np.int32)
                    label = label.astype(float_dtype)
                    
                    # Pass the config to the serialization function
                    serialized_example = serialize_example(volume, segmentation, label, self.config)
                    writer.write(serialized_example)
                except Exception as e:
                    logger.error(f"Could not process patient {patient_id} for TFRecord: {e}")