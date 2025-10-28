import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import re
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "CNN_modelTrainedPCH"
DEFAULT_IMAGE_DIR = BASE_DIR / "images"
DEFAULT_OUTPUT_CSV = BASE_DIR / "predictions.csv"

tf.compat.v1.disable_eager_execution()

def evaluate_saved_model(model_path, image_dir, output_csv=DEFAULT_OUTPUT_CSV):
    """
    Evaluate a TensorFlow SavedModel on unlabeled images and save predictions to a CSV.
    Extracts IC number from filenames (e.g., IC_39_*.png → IC = 39)
    and assigns Label = 1 if prediction >= 0.5 else 0.
    """
    model_path = Path(model_path)
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    if not model_path.exists():
        print(f"Model path does not exist: {model_path}")
        return None

    if not image_dir.exists():
        print(f"Image directory does not exist: {image_dir}")
        return None

    print(f"Loading model from: {model_path}")
    
    model = None
    try:
        sess = tf.compat.v1.Session()
        meta_graph = tf.compat.v1.saved_model.loader.load(sess, ['serve'], str(model_path))
        print("Model loaded successfully!")
        model = {'session': sess, 'meta_graph': meta_graph}
    except Exception as e:
        print(f"Loading failed: {e}")
        return None

    image_files = [f.name for f in image_dir.iterdir()
                   if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg')) and 'thresh' in f.name.lower()]

    if not image_files:
        print(f"No image files found in the directory: {image_dir}")
        return None

    df = pd.DataFrame({'filename': image_files})
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_it = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory=str(image_dir),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(270, 400),
        shuffle=False
    )

    print(f"Found {len(test_it.filenames)} test images in {image_dir}")

    predictions = []
    print("Making predictions...")

    try:
        signature = model['meta_graph'].signature_def['serving_default']
        input_name = list(signature.inputs.keys())[0]
        output_name = list(signature.outputs.keys())[0]

        input_tensor = model['session'].graph.get_tensor_by_name(signature.inputs[input_name].name)
        output_tensor = model['session'].graph.get_tensor_by_name(signature.outputs[output_name].name)

        for i in range(len(test_it)):
            batch_x = test_it[i]
            preds = model['session'].run(output_tensor, feed_dict={input_tensor: batch_x})
            preds = preds.flatten()
            predictions.extend(preds)

    except Exception as e:
        print(f"Error during prediction: {e}")
        model['session'].close()
        return None

    model['session'].close()

    # ---------------------------------------------------
    # Build results dataframe
    # ---------------------------------------------------
    results = []
    for filename, pred in zip(test_it.filenames, predictions):
        # Extract IC number using regex (e.g., IC_39_thresh.png → 39)
        match = re.search(r'IC_(\d+)', filename)
        ic_num = int(match.group(1)) if match else None
        label = 1 if pred >= 0.5 else 0
        results.append((ic_num, label))

    results_df = pd.DataFrame(results, columns=['IC', 'Label']).dropna()
    results_df = results_df.sort_values(by='IC').astype({'IC': int, 'Label': int})

    # ---------------------------------------------------
    # Save to CSV
    # ---------------------------------------------------
    output_path = output_csv if output_csv.is_absolute() else (BASE_DIR / output_csv)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(results_df.head(10))  # Show first 10 results for confirmation

    return results_df


if __name__ == "__main__":
    model_path = Path(os.environ.get("CNN_MODEL_PATH", DEFAULT_MODEL_PATH))
    image_dir = Path(os.environ.get("CNN_IMAGE_DIR", DEFAULT_IMAGE_DIR))
    output_csv = Path(os.environ.get("CNN_OUTPUT_CSV", DEFAULT_OUTPUT_CSV))

    results_df = evaluate_saved_model(model_path, image_dir, output_csv)

    if results_df is not None:
        print(f"\nModel evaluation completed successfully! Total {len(results_df)} predictions.")
    else:
        print("\nModel evaluation failed!")
