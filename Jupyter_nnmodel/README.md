# Content: Jupyter Version of 2nd place code kaggle-porto-seguro
## [Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) in Kaggle Competition

### Popose
The popose of this jupyter script is to make flow of code readable and easy to understand. Some code have been changed from the orignial author, but the concept of processing the 2nd place code is exactly same. For the some feature engineering processing, I put these codes into `feature_generater.py`

### Install

This project requires **Python 2.7 or 3.6** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [Keras](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [xgboost](https://xgboost.readthedocs.io/)
- [pickle](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [itertools](https://docs.python.org/2/library/itertools.html)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Code
1. Run `fea_eng0.py` to get your first features as a pickle file `fea0.pk`. 

Note: if you are using python 3, you need to switch the code, `fea_eng0.py`, in the last line. So the pickle file would be able to read in `nn_model.ipynb`.
 
2. the manin code is provided in the `nn_model.ipynb` notebook file. You will also be required to use the included `util.py` and `feature_generater.py` Python files, the `train.csv` and `test.csv` dataset file,which you have to download from [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data) into the input folder, to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the. During the operation of `nn_model.ipynb` in , the defualt output will be created in model folder. If you are interested in `util.py` and `feature_generater.py`, please feel free to explore these Python files. 


### Run

In a terminal or command window, navigate to the top-level project directory `Jupyter_Version/` (that contains this README) and run one of the following commands:

```bash
ipython notebook nn_model.ipynb
```  
or
```bash
jupyter notebook nn_model.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

## Data

You can download data from [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data)
In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.


