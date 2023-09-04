# DeepDicomSort_MONAI
Implementing the model from DeepDicomSort in MONAI
## Capabilities

## Limitations
1. Does no support masks in the data right now
2. Only supports DICOM datasets for now
3. Missing commentaries
4. No support for batch size > 1

## How to use the sorting script
Run the main.py file with python:
python main.py ROOT_DIR [--label-indicator LABEL_INDICATOR]

ROOT_DIR is the directoy containing the DICOM files and the label-indicator defines which DICOM attribute should be used for the generation of the regex expressions in the bidsmap ("SeriesDescription" is the default)