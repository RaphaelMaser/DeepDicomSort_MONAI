import argparse
import os
import yaml
from yaml import Loader
from tensorflow.keras.models import load_model
import pickle
from utils.data import *
import pandas as pd
import wget

def test(monai_dataset, tf_dataset, model):
    # Should be used for testing the models accuracy later on
    pass

def inference(tf_dataset, model):
    print("Beginning Inference")
    output = {"prediction": [], "label": []}
    
    for sample in tf_dataset:
        image = sample[0][0]
        extra_data = sample[0][1]
        label = sample[1]
        
        prediction = model.predict([image, extra_data])
        for o in prediction: 
            output["prediction"].append(np.argmax(o))
            output["label"].append(label)
        
    return output

def load_cached_output(path):
    
    if not os.path.exists(path):
        output = None
    else:
        print("Loading cached output")
        output = pickle.load(open(path, "rb"))
    return output

def cache_output(output, path):
    print("Caching output")
    pickle.dump(output, open(path, "wb"))


def get_majority_vote(output):
    '''
    takes the output of the inference function and returns a dataframe with the majority vote for each label
    input: dict with keys "predictions" and "labels"
    output: dict with keys "predictions" and "labels"
    '''
    
    output_df = pd.DataFrame(output)
    
    # count occurences for each label/prediction pair
    output_df = output_df.groupby(["label", "prediction"]).size().reset_index(name="count")
    
    # find prediction with highest count for each label
    output_df = output_df.loc[output_df.groupby("label")["count"].idxmax()]
    
    return output_df

def construct_regex(df):
    df = df.groupby("prediction").agg({"label": lambda x: list(x)}).reset_index()
    filter = {}
    
    for i, row in df.iterrows():
        regex = ""
        
        for key in row["label"]:
            if regex == "":
                regex += f"({key})"
            else:
                regex += "|" + f"({key})"
            
        filter[row["prediction"]] = regex
        
    return filter
    

def construct_bidsmap(output, label_name):
    print("Constructing bidsmap")
          
    # compute filter for bidsmap
    majority_vote = get_majority_vote(output)
    filter = construct_regex(majority_vote)
    
    with open('./yaml/config.yaml', 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    with open("./yaml/bidsmap_template.yaml", "r") as f:
        bidsmap_template = yaml.safe_load(f)
        
    bidsmap = bidsmap_template["bidsmap"]
    entry_templates = ["entry_templates"]
    
    for key, value in filter.items():
        key_config = config["labels"][key]["bids"]

        format = key_config["format"] # e.g. DICOM
        mod_type = key_config["type"] # e.g. anat, dwi, ...
        
        attributes = key_config.get("attributes", None) # e.g. SeriesDescription, ProtocolName, ...
        bids = key_config.get("bids", None) # e.g. run, suffix, ...
        meta = key_config.get("meta", None) # e.g. ce
        
        # returns the entry template for the given format and type (with provenance, attribute, bids and meta fields)
        entry = deepcopy(bidsmap_template["entry_templates"][format][mod_type][0])
                
        # modify attributes by the modality-specific attributes
        if attributes is not None:
            if entry.get("attributes", None) is None:
                entry["attributes"] = {}
            for attribute_key, attribute_value in attributes.items():
                entry["attributes"][attribute_key] = attribute_value
        
        # modify bids by the modality-specific bids
        if bids is not None:
            if entry.get("bids", None) is None:
                entry["bids"] = {}
            for bids_key, bids_value in bids.items():
                entry["bids"][bids_key] = bids_value

        # modify meta by the modality-specific meta
        if meta is not None:
            if entry.get("meta", None) is None:
                entry["meta"] = {}
            for meta_key, meta_value in meta.items():
                entry["meta"][meta_key] = meta_value
        
        # add regex expression to attributes
        entry["attributes"][label_name] = value
        
        # add entry to bidsmap (and create lists if necessary)
        if bidsmap[format].get(mod_type, None) is None:
            bidsmap[format][mod_type] = []
            
        bidsmap[format][mod_type].append(entry)
        
    return bidsmap
            
 # Custom representer for None values
def none_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:null', '')
    
def run(root_dir, label_indicator):
    cache_path = "cache/output.pkl"
    os.makedirs("cache", exist_ok=True)
    
    with open('./yaml/config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=Loader)
        
    # Searches for files. If dicom files are found, they are converted to nifti. Returns list of files and mode
    discovered_files, mode = discover_files(root_dir)
    
    
    # Get monai-ready list of nifti files
    data = get_data(discovered_files, label_indicator, mode)
    
    # Create transforms and datasets
    transforms = create_transforms()
    monai_dataset, tf_dataset = create_dataset(data, transforms)

    # Download models
    if not os.path.exists("trained_models/model_all_brain_tumor_data.hdf5"):
        os.makedirs("trained_models", exist_ok=True)
        print("Downloading model")
        file = wget.download(url = "https://github.com/Svdvoort/DeepDicomSort/raw/master/Trained_Models/model_all_brain_tumor_data.hdf5", out="trained_models")
        
    # Load model
    model = load_model(config["models"]["all_data"])

    # If cached output exists, load it. Otherwise, run inference
    output = load_cached_output(cache_path)
    if output is None:
        if mode == "dicom":
            output = inference(tf_dataset, model)
            cache_output(output, cache_path)
        elif mode == "bids":
            output = test(monai_dataset, tf_dataset, model)
        else:
            raise Exception("Unknown mode")
    
    bidsmap = construct_bidsmap(output, label_indicator)

    # Add the custom representer to the SafeDumper class
    yaml.add_representer(type(None), none_representer, Dumper=yaml.SafeDumper)

    if not os.path.exists("output"):
        os.makedirs("output", exist_ok=True)
        
    with open("output/bidsmap.yaml", "w") as f:
        yaml.dump(bidsmap, f, default_flow_style=False, Dumper=yaml.SafeDumper)
        
    return output
    