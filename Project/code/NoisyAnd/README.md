# Summary of Files
 - train_and_predict_orig_v2.py: Train Model
 - predict_v2.ipynb: Run Evaluation (get accuracy)
 - predict_get_heatmap.ipynb: Generate heatmap from model

# Location of Data



# How to Run

For all files:

1. Make sure libraries below are installed:
 - tensorflow
 - keras
 - numpy
 - cv2
 - matplotlib
 - tf_cnnvis

2. Modify 'input_snippet_training_dir', 'input_snippet_validation_dir', 'input_snippet_test_dir' to point to the correct training, validation and test directory respectively.

For train_and_predict_orig_v2.py,

3. set 'SAVE_DIRECTORY' and make sure this directory exists.

For predict_get_heatmap.ipynb and predict_v2.ipynb:

4. Make sure 'SAVE_DIRECTORY' and 'MODEL_NAME' points to the model you saved.

For predict_get_heatmap.ipynb:

5. Set 'heatmap_output_folder' and make sure the directory exists. 
