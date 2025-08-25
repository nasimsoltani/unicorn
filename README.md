# ORAN

This is the code and dataset repository for UNICORN paper in ICMLCN 2025.

## Dataset
The traces that are used for training/test the models are included in the `trace` folder in the git repository.
Full dataset before and after Colosseum can be found [here](https://utexas.box.com/s/rvtfrv09gnaf1xqaj0a1bfcdvsj0p3n1).

## Running the code

### Preprocessing
As the first step you need to preprocess the dataset using `preprocessing.py`. Run it as:
'''
python preprocessing.py 
'''
The above script reads the KPI traces stored in the `.csv` files in the `trace` directory, preprocesses them and saves training, validation, and test partitions and statistics in the `.pkl` files stored in `pkls` directory.

### Training and Test pipeline
Training and test pipeline can be run only after **Preprocessing** step is done.

The pipeline can be run through `run_pipeline.sh` with the following content.

```
python top.py \
--gpu_id $1 \
--Train \
--Test \
--OOD_class 'Teams' \
--model_flag 'ResConvNet' \
--weight_filename 'weights-ICMLCN4-ResConvNet-5ids-oodindex4.pt'
```

This bash file runs the file `top.py` that calls the pipeline for training and test. Here is a short description of the arguments.

- `--gpu_id`: The ID of GPU node on your system. This can be set to 0 or 1 or 2 or ... depending on the GPU node you want to use.
- `--Train`: store_true parameter which indicates if we want to train an NN model.
- `--Test`: store_true parameter which indicates if we want to test a trained NN model.
- `--OOD_class`: The class name you would like to exclude from the training set. Can be any of the following: 'CallofDuty', 'Facebook', 'Meet', 'Zoom', 'Teams', 'Twitch'
- `--model_flag`: Indicating the model architecture. Can be 'ResConvNet' or ForwConvNet
- `--weight_filename`: Indicates the filename of NN weight files, that is used only for testing the trained model.

The pipeline can be run through:
```
./run_pipeline.sh 1
```
Where `1` is the GPU ID.

The test pipeline saves model predictions in `.pkl` files in the `results` directory.

### Plotting the results
