import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils.helpers import load_npy
from utils.augmentations import get_augmenter
from data.slicer import Slicer

class BratsDataset3D:
    """
    TensorFlow Dataset loader for 3D BraTS data from TFRecords.
    """
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.tfrecord_path = os.path.join(config['processed_data_path'], 'tfrecords')
        self.augmenter = get_augmenter(config) if mode == 'train' else None
        
        filename = f"{mode}.tfrecord"
        self.filepath = os.path.join(self.tfrecord_path, filename)
        
        # Count the number of records in the TFRecord file to determine dataset size
        self.dataset_size = sum(1 for _ in tf.data.TFRecordDataset(self.filepath))

    def _parse_tfrecord_fn(self, example):
        feature_description = {
            'volume': tf.io.FixedLenFeature([], tf.string),
            'segmentation': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        
        volume = tf.io.parse_tensor(example['volume'], out_type=tf.float32)
        
        # --- THIS IS THE FIX ---
        # Match the data type used when writing the TFRecord (int32)
        segmentation = tf.io.parse_tensor(example['segmentation'], out_type=tf.int32)
        # ----------------------

        label = tf.io.parse_tensor(example['label'], out_type=tf.float32)

        # Set shape explicitly. This is crucial for model building.
        vol_shape = self.config['volume_shape']
        volume = tf.reshape(volume, vol_shape)
        segmentation = tf.reshape(segmentation, vol_shape[:-1])


        if self.config['output_type'] == 'benign_vs_malignant':
            label = tf.reshape(label, [1])
            return volume, label
        else: # segmentation
            seg_cat = tf.keras.utils.to_categorical(segmentation, num_classes=self.config['num_classes'])
            return volume, seg_cat

    def get_dataset(self, batch_size):
        dataset = tf.data.TFRecordDataset(self.filepath, num_parallel_reads=tf.data.AUTOTUNE)
        
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=100) # Shuffle records

        dataset = dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if self.mode == 'train' and self.augmenter:
            dataset = dataset.map(self.augmenter.augment, num_parallel_calls=tf.data.AUTOTUNE)
        
        if self.mode != 'test':
             dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

class BratsDataset2D:
    """
    TensorFlow Dataset loader for 2D sliced BraTS data.
    Uses tf.data.Dataset.interleave for memory-efficient loading.
    """
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.slicer = Slicer(axis=config['slice_axis'])
        self.data_path = config['processed_data_path']
        self.patient_ids = [p for p in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, p))]
        self.augmenter = get_augmenter(config) if mode == 'train' else None

        # Split patient IDs for training, validation, and testing
        train_val_ids, test_ids = train_test_split(self.patient_ids, test_size=config['train_val_test_split'][2], random_state=42)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=config['train_val_test_split'][1] / (1 - config['train_val_test_split'][2]), random_state=42)

        if mode == 'train':
            self.ids_to_load = train_ids
        elif mode == 'validation':
            self.ids_to_load = val_ids
        else: # test
            self.ids_to_load = test_ids

        self.dataset_size = len(self.ids_to_load) # Size is now the number of patients

    def _slice_generator(self, patient_id_tensor):
        """A generator that yields slices from a single patient volume."""
        patient_id = patient_id_tensor.decode('utf-8')
        patient_dir = os.path.join(self.data_path, patient_id)
        
        try:
            volume = load_npy(os.path.join(patient_dir, 'volume.npy'))
            segmentation = load_npy(os.path.join(patient_dir, 'segmentation.npy'))
            
            # For segmentation, only process patients with tumors
            if self.config['output_type'] == 'segmentation' and np.sum(segmentation) == 0:
                return

            sliced_data = self.slicer.slice_volume(volume, segmentation)
            
            for img_slice, seg_slice in sliced_data:
                img_slice_f32 = img_slice.astype(np.float32)
                seg_slice_cat = tf.keras.utils.to_categorical(
                    seg_slice, num_classes=self.config['num_classes']
                ).astype(np.float32)
                yield img_slice_f32, seg_slice_cat
        except FileNotFoundError:
            return

    def get_dataset(self, batch_size):
        """Builds the final tf.data.Dataset pipeline."""
        patient_dataset = tf.data.Dataset.from_tensor_slices(self.ids_to_load)

        if self.mode == 'train':
            patient_dataset = patient_dataset.shuffle(self.dataset_size)

        dataset = patient_dataset.interleave(
            lambda patient_id: tf.data.Dataset.from_generator(
                self._slice_generator,
                output_signature=(
                    tf.TensorSpec(shape=self.config['input_shape'], dtype=tf.float32),
                    tf.TensorSpec(shape=(*self.config['input_shape'][:2], self.config['num_classes']), dtype=tf.float32)
                ),
                args=(patient_id,)
            ),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=256)
            if self.augmenter:
                dataset = dataset.map(self.augmenter.augment, num_parallel_calls=tf.data.AUTOTUNE)

        if self.mode != 'test':
            dataset = dataset.repeat()
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
