### Code for Dligach and Miller, 2018 *SEM paper *Learning Patient Representations from Text*

To train a billing code prediction model:

* extract CUIs from MIMIC III patient data
* cd Codes
* ft.py cuis.cfg.

To run the experiments with i2b2 data:

* cd Comorbidity
* svm.py sparse.cfg
* svm.py dense.cfg
