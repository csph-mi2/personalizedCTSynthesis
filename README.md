# Repository for longitudinal ct image synthesis
This is the repository for the: Population-Driven Synthesis of Personalized Cranial Development from Cross-Sectional Pediatric CT Images.

This repository contains the scripts to make random synthesis across time span and to make longitudinal predictions. 

The Templates folder contains template head and bone segmentation images and landmarks for preprocessing purposes.

## Dependencies:
The main dependencies for this repository are the following:
- [Python](python.org)
- [NumPy](https://numpy.org/install/)
- [SimpleITK](https://simpleitk.org/)
- [PyTorch](https://pytorch.org/)
- [VTK](https://vtk.org/)
- [scipy](https://scipy.org/)

Please, install all packages and their dependencies using the included Requirements.txt file:
```
pip install -r Requirements.txt
```

## Using the code

### Quick summary
**Inference.py** makes longitudinal predictions based on input CT (mha format), cranial base landmarks (VTK polydata format) and desired target age and sex.
- Input: ctImage.mha, CranialBaseLandmarks.vtp, age (normalized to 0-1, with 1 representing 10 years assuming 0 is the time of birth), sex (1 for female and 0 for male)
    - Adjust the age to the desired target age (line 35).
    - Adjust the sex to the desired target sex (line 36).

- Output: .mha image for the prediction (``prediction.mha``).
