# dldoc
Deep Learning based Cardiology AI to predict Atrial Fibrillation in ECG data

## Authors

Florian Schroevers  
Abel Oakley

## Files

### `global_params.py`

	Description:
		Loads the parameters set in the 'config.json' file.

---

### `core.py`

	Description:
    	The main script that runs all required code to load the data, pre-process the data,
        create the model, trains the network and show the metrics.

Functions:

* `main`

		Description:
			Runs the whole process as specified above
		Args:
		Returns:

---

### `main.py`

    Description:
        Loads the model and gives prediction for an ecg selected by the user

Functions:

* `main`

        Description:
            Runs the whole process as specified above
        Args:
        Returns:

---

### `data_generator.py`

	Description:
    	File for handling the loading of ecg data into a usable format


Functions:


* `get_ids`:

		Description:
        	returns a list of all patient id's in the dataset
        Args:
        Returns:
          ids : list
              a list of all patient id's in the dataset
* `get_ecg_by_id`:

		Description:
        	returns the contents of an ecg file and its target (rythm), given a data id.
            if the time is not given, a random ecg will be chosen.
        Args:
            patient_id : str
                the id of the patient
            t : str or Nonetype [optional, default: None]
                the time at which the ecg was taken
        Returns:
            tuple (5000x8 np.array, int): a numpy array with the data and the target
                                      (0 if sinus rythm, 1 otherwise)
* `get_ecg_fnames`:

		Description:
        	returns a list of filenames which are ecg's for a given patient
        Args:
            patient_id : str
                the id of the patient
        Returns:
            fnames : list
                a list containing all filenames with ecg's of this patient
* `get_times`:

		Description:
        	returns a list of times at which ecg's for a given patient are taken
        Args:
            patient_id : str
                the id of the patient
        Returns:
            times : list
                a list containing times at which ecg's of this patient are taken
* `get_gender`:

		Description:
        	returns a the gender of the given patient
        Args:
            patient_id : str
                the id of the patient
        Returns:
            gender : str
                'M' if the patient is male, 'F' if female
* `get_time_rythm_mapping`:

		Description:
        	returns a dict mapping the times to the rythm at that time for a given patient
        Args:
            patient_id : str
                the id of the patient
        Returns:
            time_rythm_dict : dict
                keys are times ecg's were taken, values are rythms at those times
* `get_time_fname_mapping`:

		Description:
        	returns a dict mapping the times to the filename of the ecg taken at
            that time for a given patient
        Args:
            data_id : str
                the id of the patient
        Returns:
            times_fnames_dict : dict
                keys are times ecg's were taken, values are filenames of
                ecg's taken at those times
* `get_feat_data`:

		Description:
        Args:
        	df: a pandas dataframe with atleast 7 columns, 6 of which are not params
        Returns:
            features: part of the dataframe that makes up the processed features
                    Shape is 6 + number of params by number of ECG's
            targets: targets that belong to those features
                    Shape is 1 by number of ECG's
* `get_data`:

		Description:
        	returns data in the directory specified in the helpers.py file
        Args:
            n_files : (Nonetype or int) [optional, default: None]
                the number of samples to return, return all available data if set to
                None
            extension : str [optional, default: '.csv']
                the extension (filtype) of the data. can be anything, as long as
                it's readable by np.loadtxt
            split : (bool or str) [optional, default: False]
                to split data 50/50 into healthy/non-healthy or not (only works if
                target is set to None)
                if set to 'max', the function will determine what the max amount of
                files is while keeping the ration 50/50 (will override n_files)
            channels : (Nonetype or np.array) [optional, default: None]
                indices of channels to return or None for all channels
            norm : (bool) [optional, default: False]
                normalize the channels
            exclude_targets : (list) [optional, default: []]
                a list of conditions not to return (0: healthy, 1: afib, 2: afl,
                3: svt, 4: unknown)
            return_fnames : bool [optional, default: False]
                wheter to return a the filenames of the data
            randomize_order : bool [optional, default: True]
                whether to randomize the order of the data
            n_points : int [optional, default: 5000]
                the number of data points to exctract
        Returns:
            data_x : np.ndarray
                the ecg data itself as a 3D array with shape
                (n_ecgs, ecg_len, n_channels)
            data_y : np.ndarray
                an array of target variables
            files : list [optional]
                a list of all files

---

### `data_preprocessing.py`


	Description:
    	Handles the preprocessing of data (smoothing etc)


Functions:
* `savitzky_golay`:

		Description:
        	Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
            The Savitzky-Golay filter removes high frequency noise from data.
            It has the advantage of preserving the original shape and
            features of the signal better than other types of filtering
            approaches, such as moving averages techniques.
            Parameters
        Args:
        	y : array_like, shape (N,)
       			the values of the time history of the signal.
            window_size : int
                the length of the window. Must be an odd integer number.
            order : int
                the order of the polynomial used in the filtering.
                Must be less then `window_size` - 1.
            deriv: int
                the order of derivative to compute (default = 0 means only smoothing)
        Returns:
        	ys : ndarray, shape (N)
        		the smoothed signal (or it's n-th derivative).

* `cn`:

		Description:
        	perform fourier series analysis on a signal and return the nth coefficient
            for a function that approximate the wave

            assumes the data is modelled by an even function (f(x) = f(-x)).
            the function on this page is followed (the second one under cosine series)
            https://en.wikipedia.org/wiki/Fourier_sine_and_cosine_series
        Args:
            signal : np.ndarray
                a 1d array containing the signal to perform the analysis on
            n : int
                the nth coefficient to return.
        Returns:
            c : int
                the nth coefficient that approximate the given data

* `get_fourier_coefficients`:

		Description:
        	uses the function cn (see above) to get an array of coefficients that
            approximate a given signal
            the function that reconstructs this wave is: reconstruct_wave
        Args:
            signal : np.ndarray
                a 1d array containing the signal to perform the analysis on
            resolution : int [optional, default: 600]
                the amount of coefficients to return. the higher the resolution the
                higher the fidelity of the reconstructed wave
        Returns:
            coefficients : np.ndarray
                a 1d array containing the coefficients that model the given data

* `reconstruct_wave`:

		Description:
        	reconstructs a wave using a given set of coefficients using the function as
            seen on this page (first equation under cosine series)
            https://en.wikipedia.org/wiki/Fourier_sine_and_cosine_series
        Args:
            coefficients : np.ndarray
                a 1d array of coefficients to reconstruct wave from, such as those
                given by the function get_fourier_coefficients
            signal_length : int
                the target length of the reconstructed array
        Returns:
            reconstruction : np.ndarray
                a 1d array of length signal_length that approximates the data of
                which the coefficients were extracted

* `fourier_straighten`:

		Description:
        	straighten an ecg using a low resolution to obtain ecg baseline, and then
    		straightening the ecg by setting the baseline to zero
        Args:
            signal : np.ndarray
                the ecg signal to straighten
            resolution : int [optional, default: 20]
                the resolution of the baseline. if set to high important details
                from the ecg will be removed, recommended to keep under 30
        Returns:
            corrected_ecg : np.ndarray
                the straightened array

* `preprocess_data`:

		Description:
        	preprocess the data by smoothing and straightening.
        Args:
            data_x : np.ndarray
                the data to preprocess.
        Returns:
            p_data_x : np.ndarray
                preprocessed data

* `save_data`:

		Description:
        	saves the preprocessed data to the location specified in global_params.py
            saves the files as the original filename + 'preprocessed' if the oriinal
            filenames are given, otherwise as a generic name with target variable in it
        Args:
            data_x : np.ndarray
                the ecg data to save
            data_y : np.ndarray
                the targetsof the ecg's
            fnames : list [optional, default: []
                the filenames of the original files
        Returns:

* `pulse_scale`:

		Description:
        	scales an array to a given length, using 1d linear interpolation
        Args:
            pulse : np.ndarray
                the array to scale
            target_size : int
                the size to scale to
        Returns:
            scaled_pulse : np.ndarray
                the scaled pulse

* `extract_windows`:

		Description:
        	extract all pulses from an ecg and scale them to a given size
        Args:
            data_x : np.ndarray
                an array of ECG's
            data_y : np.ndarray
                an array of targets of the ECG's
            pulse_size : int [optional, default: 80]
                the size to scale the pulses to
        Returns:
            pulse_data_x : np.ndarray
                an array of pulses
            pulse_data_y : np.ndarray
                an array of targets of the corresponding pulses

---

### `feature_extraction.py`


	Description:
    	Handles the extraction of features from the data


Functions:
* `get_rpeaks`:

		Description:
        	returns an array of indices of the r peaks in a given ecg
        Args:
            ecg : np.ndarray
                an ecg
        Returns:
            rpeaks : np.ndarray
                an array of indices of the r peaks

* `get_peak_offset`:

		Description:
        	calculate the mean of deviations from the horizontal location
    		of r peaks of an ecg to their expected position
        Args:
            ecg : np.ndarray
                the ecg
            rpeaks : np.ndarray
                an array containing the indices of the r peaks
        Returns:
            peakdev : int
                The mean of deviations from the r peaks to their expected location
                if spread uniformly over the ecg

* `get_heartrate`:

		Description:
        	calculate the heartrate of an ecg
        Args:
            rpeaks : np.ndarray
                an array containing the indices of the r peaks
        Returns:
            hr : int
                the heartrate of an ecg

* `get_ppeaks`:

		Description:
        	Detects P-peaks in an ECG by looking for max values inside windows between
            the R-peaks, but takes the T-peak into regard as well as missing P-peaks.
            returns the amount of P-peaks and a number between 0 and 1 representing the
            relation between the mean of the P-peaks and the mean of the R-peaks.
        Args:
            data_x : np.ndarray
                3D array with ECG data (should be smoothed)

        Returns:
            mean_peak_height : 1D array
                array of the means of the p-peaks in ECGs
            n_ppeaks : 1D array
                array of the number of p-peaks in ECGs

* `show_correlation_heatmap`:

		Description:
        	shows a correlation heatmap of a given dataframe
        Args:
            df : pd.DataFrame
                the dataframe to calculate the correlation matrix of

* `load_extracted_features`:

		Description:
        	load the extracted features from the location specified in global_params.py
        Args:
            fname : str
                the name of the file to load
        Returns:
            df : pd.DataFrame
                a dataframe with the features

* `extract_features`:

		Description:
        	creates a dataframe from a given collection of ecgs containing the fourier
    		series coefficients, heartrate and peak offset
        Args:
            data_x : np.ndarray
                a set of ecgs. must only contain one channel per ecg. can be 1d, 2d
                or 3d
            data_y : np.ndarray
                the targets of the ecg's
            smooth : bool [optional, default: False]
                whether to smooth the data
            fourier_resolution : int [optional, default: 600]
                the amount of coefficients to return. the higher the resolution the
                higher the fidelity of the reconstructed wave
            save_to_file : str [optional, default: '']
                saves to this file if the length of this string > 0, with this as
                filename
        Returns:
            df : pd.DataFrame
                a dataframe with the extracted features

---

### `neural_network.py`


	Description:
    	Implements the neural network model as well as the fitting and evaluation of the model


Functions:
* `prepare_train_val_data`:

		Description:
        	splits the data in a training, validation and test set, while maintaining a
            50/50 split of targets in all sets, so that the network won't learn to
            always predict one target
        Args:
            data_x : np.ndarray
                an array of input data
            data_y : np.ndarray
                an array of targets of the data
            feature_data : pandas.DataFrame
                a dataframe with any additional extracted features
            tvt_split : list
                a list with three floats that represent the fraction of the size of
                the training, validation and test (tvt) sets respectively
            equal_split_test : bool
                whether to split the test set 50/50
        Returns:
            train_x : dict
                a dict containing the data of this set with input name as key and
                data as value
            train_y : np.ndarray
                an array with targets for this set
            validation_x : dict
                a dict containing the data of this set with input name as key and
                data as value
            validation_y : np.ndarray
                an array with targets for this set
            test_x : dict
                a dict containing the data of this set with input name as key and
                data as value
            test_y : np.ndarray
                an array with targets for this set

* `ffnet`:

		Description:
        	returns model
        Args:
            ecg_shape : tuple
                the shape of the input
            summarize : bool [optional, default:False]
                whether to show a summary of the model
        Returns:
            model : keras.models.Model
                the model

* `precision`:

		Description:
        	Precision metric, only computes a batch-wise average of precision.

        	Computes the precision, a metric for multi-label classification of
			how many selected items are relevant.

* `recall`:

		Description:
			Recall metric, only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.

* `train`:

		Description:
        	fit data on a model and return the trained model
        Args:
            model : keras.models.Model
                the model to evaluate
            x_train : dict
                a dictionary mapping input names to actual data
            y_train : np.ndarray
                the targets of the train data
            x_val : dict
                a dictionary mapping input names to actual data
            y_val : np.ndarray
                the targets of the validation data
            batch_size : int [optional, default: 32]
                the size of the batches to be fed into the network
            epochs : int [optional, default: 32]
                the number of epochs (times to run the network)
            save : bool [optional, default: False]
                whether to train the saved network
        Returns:
            r : list
                list of the loss and metrics specified by the model after running
                the model on the test data

* `eval`:

		Description:
        	evaluate the model on a test set (consisting of pulses)
        Args:
            model : keras.models.Model
                the model to evaluate
            x_test : dict
                a dictionary mapping input names to actual data
            y_test : np.ndarray
                the targets of the test data
            batch_size : int [optional, default: 32]
                the size of the batches to be fed into the network
        Returns:
            r : list
                list of the loss and metrics specified by the model after running
                the model on the test data

---

### `helpers.py`


	Description:
    	Helper functions to be used by multiple files in the project


Functions:
* `progress_bar`:

		Description:
        	prints the current state of the progress
---
