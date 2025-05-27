

Your task is to extract data segements for intput to train a 1D convolutional neural net as a REM classifier.

You have 20 nights worth of sleep data. they are in the folder 'provided_data' and labeled 'night_XX_label.csv' and 'night_XX.edf' where XX is the number(01, 02, 03, ..., 20).

The label contains one of 'Wake', 'Light', 'Deep', or 'REM' for each timestamp. We only care about 'REM'.

You should preprocess the data as necessary (apply a notch pass and band pass filter). After,  split the edf file into 15-second time intervals with a clear label. For each window, the label should be 1 if the last 2 seconds of the window are all REM and 0 otherwise. Name the output file extracted_REM_windows.

Try to get as many positive cases as possible from each REM window. You may overlap as necessary.

Build the data extraction pipeline in the file extract_rem_data_for_training.py.

I have attached another file, alertness_model_training.ipynb, for your reference. Be warned that this is a large file with a lot of miscelanious information, but it may be useful as a reference for how to handle the edf files.





