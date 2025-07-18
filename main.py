import argparse
import os
import yaml
import numpy as np
import time
import datetime

from utils.helpers import load_config, load_npy, ensure_dir
from utils.logger import get_logger
from data.brats2020_preprocess import Preprocessor
from data.tfrecord_writer import TFRecordWriter
from training.trainer2d import Trainer2D
from training.trainer3d_cls import Trainer3DClassifier
from training.trainer3d_seg import Trainer3DSegmentation
from inference.predict2d import Predictor2D
from inference.predictor_3d import Predictor3D
from inference.predictor_3d_cls import Predictor3DClassifier
from inference.reconstruct3d import Reconstructor3D
from inference.visualizer import Visualizer
from evaluation.evaluator import Evaluator

logger = get_logger("BraTS_Unet_Project")

def load_and_prepare_config(args):
    """Loads base config and merges profile settings."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    profile_name = config.get('active_profile', 'gpu_high_mem')
    profile_settings = config.get(profile_name, {})
    
    # Merge profile settings into the main config
    config.update(profile_settings)

    # Override from command-line arguments
    if args.model:
        config['model'] = args.model
    if args.output_type:
        config['output_type'] = args.output_type
    
    return config

def main(args):
    config = load_and_prepare_config(args)
    logger.info(f"----- Using Configuration -----")
    logger.info(yaml.dump(config, indent=2))

    if args.mode == 'preprocess':
        logger.info("----- Running Preprocessing -----")
        preprocessor = Preprocessor(config=config)
        preprocessor.run()

    elif args.mode == 'convert_to_tfrecord':
        logger.info("----- Converting .npy to .tfrecord -----")
        writer = TFRecordWriter(config)
        writer.convert()

    elif args.mode == 'train':
        logger.info(f"----- Running Training for model: {config['model']} -----")
        if config['model'] == 'UNET2D':
            trainer = Trainer2D(config)
        elif config['model'] == 'Classifier3D':
            trainer = Trainer3DClassifier(config)
        elif config['model'] == 'UNET3D':
            trainer = Trainer3DSegmentation(config)
        else:
            raise ValueError(f"Unknown model type for training: {config['model']}")
        trainer.train()

    elif args.mode == 'predict':
        logger.info(f"----- Running Prediction for model: {config['model']} -----")
        if not args.patient_id:
            raise ValueError("Patient ID must be provided for prediction mode.")

        patient_data_path = os.path.join(config['processed_data_path'], args.patient_id)
        if not os.path.exists(patient_data_path):
             raise FileNotFoundError(f"Processed data for patient {args.patient_id} not found.")

        volume = load_npy(os.path.join(patient_data_path, 'volume.npy'))
        ground_truth_seg = load_npy(os.path.join(patient_data_path, 'segmentation.npy'))
        
        visualizer = Visualizer(config)
        flair_volume = volume[..., 3] # Use FLAIR for background visualization

        if config['model'] == 'UNET2D':
            predictor = Predictor2D(config, model_path='unet2d_best.keras')
            predicted_slices, mri_slices = predictor.predict_volume(volume)

            reconstructor = Reconstructor3D(config)
            reconstructed_vol = reconstructor.stack_slices(predicted_slices, volume.shape)
            post_processed_vol = reconstructor.post_process(reconstructed_vol)
            
            logger.info("Generating slice visualizations...")
            for i in range(len(predicted_slices)):
                 if np.sum(ground_truth_seg[..., i]) > 0:
                    visualizer.plot_slice_comparison(args.patient_id, mri_slices[i], ground_truth_seg[..., i], np.argmax(predicted_slices[i], axis=-1), i)
            
            visualizer.plot_3d_reconstruction(args.patient_id, flair_volume, ground_truth_seg, post_processed_vol)

        elif config['model'] == 'UNET3D':
            predictor = Predictor3D(config, model_path='unet3d_seg_best.keras')
            predicted_seg = predictor.predict_volume(volume)
            post_processed_vol = Reconstructor3D(config).post_process(predicted_seg)
            
            logger.info("Generating 3D visualizations...")
            visualizer.plot_3d_reconstruction(args.patient_id, flair_volume, ground_truth_seg, post_processed_vol)
        
        elif config['model'] == 'CLASSIFIER3D':
            logger.info(f"Running prediction for patient: {args.patient_id}")
            
            # Instantiate the predictor
            predictor = Predictor3DClassifier(config, model_path='classifier3d_best.keras')
            
            # Load the ground truth label
            true_label_arr = load_npy(os.path.join(patient_data_path, 'label.npy'))
            true_class = int(true_label_arr[0])

            # Perform prediction and generate all outputs
            predictor.predict_and_visualize(volume, args.patient_id, true_class)

        else:
            logger.warning(f"Prediction for {config['model']} is not fully implemented in this script.")


    elif args.mode == 'evaluate':
        logger.info(f"----- Running Evaluation for model: {config['model']} -----")
        evaluator = Evaluator(config)
        if config['model'] == 'UNET2D':
            evaluator.evaluate_2d_model('unet2d_best.keras')
        elif config['model'] == 'UNET3D':
            evaluator.evaluate_3d_model('unet3d_seg_best.keras')
        elif config['model'] == 'CLASSIFIER3D':
            evaluator.evaluate_3d_classifier_model('classifier3d_best.keras')
        else:
            logger.error(f"Evaluation not implemented for model type: {config['model']}")

if __name__ == '__main__':
    # Start time for overall script
    overall_start_time = time.time()

    parser = argparse.ArgumentParser(description="BraTS 2020 Tumor Analysis Pipeline")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'convert_to_tfrecord', 'train', 'predict', 'evaluate'],
                        help="Pipeline mode to run.")
    parser.add_argument('--model', type=str,
                        help="Specify which model to use (overrides config). E.g., UNET2D, Classifier3D")
    parser.add_argument('--output_type', type=str,
                        help="Specify the output type for the model (overrides config). E.g., benign_vs_malignant, segmentation")
    parser.add_argument('--patient_id', type=str,
                        help="Patient ID for prediction mode.")

    args = parser.parse_args()
    
    main(args)
    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time

    # Format for readability
    total_time_formatted = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTotal Time taken - completed in {total_time_formatted} (HH:MM:SS)")
