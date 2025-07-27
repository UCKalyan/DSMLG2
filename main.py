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
# NEW: Import EnsemblePredictor (if you decide to use it separately in main, but evaluator will handle it)
# from ensemble_predictor import EnsemblePredictor # Only if you call it directly here


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
    # Set model_type in config based on chosen model for EnsemblePredictor
    if config['model'] == 'UNET2D':
        config['model_type'] = '2d'
    elif config['model'] == 'UNET3D':
        config['model_type'] = '3d'
    # No 'model_type' for Classifier as ensemble evaluation is for segmentation
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
            # predicted_slices will now contain the thresholded masks from Predictor2D
            predicted_slices, mri_slices = predictor.predict_volume(volume) 

            reconstructor = Reconstructor3D(config)
            # stack_slices now expects already thresholded masks (from predicted_slices)
            reconstructed_vol = reconstructor.stack_slices(predicted_slices, volume.shape) 
            post_processed_vol = reconstructor.post_process(reconstructed_vol)
            
            logger.info("Generating individual and composite slice visualizations from post-processed volume...")
            
            # Map string axis to integer index
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            slice_axis_str = config['slice_axis'].lower() # Get the string, convert to lowercase for robustness
            if slice_axis_str in axis_map:
                slice_axis_int = axis_map[slice_axis_str]
            else:
                # Fallback if config has an integer directly, or raise error if invalid
                try:
                    slice_axis_int = int(slice_axis_str)
                except ValueError:
                    raise ValueError(f"Invalid slice_axis in config: {config['slice_axis']}. Must be 'x', 'y', 'z' or an integer (0, 1, 2).")

            num_slices = post_processed_vol.shape[slice_axis_int] # Use the determined slice axis with integer index

            for i in range(num_slices):
                # Extract slices for GT and Pred from the 3D volumes
                # Ensure slicing uses integer slice_axis_int
                if slice_axis_int == 2: # Z-axis (last dimension for BraTS data)
                    gt_slice_i = ground_truth_seg[..., i]
                    pred_slice_i = post_processed_vol[..., i]
                elif slice_axis_int == 1: # Y-axis
                    gt_slice_i = ground_truth_seg[:, i, :]
                    pred_slice_i = post_processed_vol[:, i, :]
                else: # slice_axis_int == 0 (X-axis)
                    gt_slice_i = ground_truth_seg[i, :, :]
                    pred_slice_i = post_processed_vol[i, :, :]
                    
                mri_slice_i = mri_slices[i] # mri_slices are already prepared by Predictor2D's slicer

                if np.sum(gt_slice_i) > 0: # Only plot slices with ground truth tumor
                    visualizer.plot_slice_comparison(args.patient_id, mri_slice_i, gt_slice_i, pred_slice_i, i)
                    visualizer.plot_slice_composite_comparison(args.patient_id, mri_slice_i, gt_slice_i, pred_slice_i, i)
            
            logger.info("Generating 3D visualizations for individual components and composite regions...")
            visualizer.plot_3d_individual_reconstruction(args.patient_id, flair_volume, ground_truth_seg, post_processed_vol)
            visualizer.plot_3d_composite_reconstruction(args.patient_id, flair_volume, ground_truth_seg, post_processed_vol)


        elif config['model'] == 'UNET3D':
            predictor = Predictor3D(config, model_path='unet3d_seg_best.keras')
            # predicted_seg will now be the thresholded and potentially resized mask
            predicted_seg = predictor.predict_volume(volume) 
            post_processed_vol = Reconstructor3D(config).post_process(predicted_seg) # Still apply post-processing

            logger.info("Generating 3D visualizations for individual components and composite regions...")
            visualizer.plot_3d_individual_reconstruction(args.patient_id, flair_volume, ground_truth_seg, post_processed_vol)
            visualizer.plot_3d_composite_reconstruction(args.patient_id, flair_volume, ground_truth_seg, post_processed_vol)

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

    # NEW: Mode to run prediction on the entire dataset and save to CSV
    elif args.mode == 'predict_dataset':
        logger.info(f"----- Running Dataset Prediction for model: {config['model']} -----")
        if config['model'] != 'CLASSIFIER3D':
            logger.error(f"Dataset prediction mode is only implemented for CLASSIFIER3D, but model is {config['model']}.")
            return
        
        # Instantiate the predictor
        predictor = Predictor3DClassifier(config, model_path='classifier3d_best.keras')

        # Run prediction on the test dataset and generate the report and CSV file.
        # To run on the validation set, change the argument to dataset_mode='validation'
        predictor.predict_dataset(dataset_mode='test')

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

        # --- NEW ENSEMBLE EVALUATION MODE ---
    elif args.mode == 'ensemble_evaluate':
        logger.info(f"----- Running Ensemble Evaluation -----")
        evaluator = Evaluator(config)
        
        # Determine which set of model paths to use based on the 'model' config
        if config['model'] == 'UNET2D':
            ensemble_model_paths = config.get('ensemble_model_paths_2d', [])
        elif config['model'] == 'UNET3D':
            ensemble_model_paths = config.get('ensemble_model_paths_3d', [])
        else:
            raise ValueError(f"Ensemble evaluation is only supported for UNET2D or UNET3D models. Configured model: {config['model']}")

        if not ensemble_model_paths:
            logger.error(f"No ensemble model paths defined in config for model type: {config['model']}")
            return

        evaluator.evaluate_ensemble(ensemble_model_paths)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    # Start time for overall script
    overall_start_time = time.time()

    parser = argparse.ArgumentParser(description="BraTS 2020 Tumor Analysis Pipeline")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'convert_to_tfrecord', 'train', 'predict', 'evaluate', 'predict_dataset', 'ensemble_evaluate'],
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