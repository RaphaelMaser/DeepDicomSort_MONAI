import argparse
from utils.predict import run

def main():
    parser = argparse.ArgumentParser(prog="DeepDicomSort_MONAI", description='Bidscoin regex finder for DICOM datasets')
    parser.add_argument("root_dir", type=str, help="Root directory of the dataset")
    parser.add_argument("--label_indicator", dest="label_indicator", default="SeriesDescription", type=str, help="Label indicator for the dataset, default is 'SeriesDescription'")
    args = parser.parse_args()
    root_dir = args.root_dir
    label_indicator = args.label_indicator
    
    run(root_dir, label_indicator)

main()