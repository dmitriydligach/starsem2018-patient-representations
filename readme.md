### Code for Dligach and Miller, 2018 *SEM paper *Learning Patient Representations from Text*

To train a billing code prediction model:

* extract CUIs from MIMIC III patient data
* cd Codes
* ft.py cuis.cfg.

To run the experiments with i2b2 data:

* cd Comorbidity
* svm.py sparse.cfg
* svm.py dense.cfg

For the experiments described in the paper, we used NumPy 1.13.0, scikit-learn 0.19.1, and Keras 2.0.4 with Theano 0.9.0 backend. Titan X GPU we used for training neural network models was provided by NVIDIA.
