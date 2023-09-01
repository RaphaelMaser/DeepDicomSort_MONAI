import argparse
import os
import yaml
from yaml import Loader
from tensorflow.keras.models import load_model
import pickle
from utils.data import *
import pandas as pd

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
            output["label"].append(label.numpy())
        
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
    
    # compute filter for bidsmap
    majority_vote = get_majority_vote(output)
    filter = construct_regex(majority_vote)
    
    with open('./config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=Loader)
    
    
    
    
    pass
  
def run(root_dir, label_indicator):
    cache_path = "cache/output.pkl"
    os.makedirs("cache", exist_ok=True)
    
    with open('./yaml/config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=Loader)
        
    discovered_files, mode = discover_files(root_dir)
    
    transforms = create_transforms()
    
    data = get_data(discovered_files, label_indicator, mode)
    monai_dataset, tf_dataset = create_dataset(data, transforms)

    model = load_model(config["models"]["train_data"])
    #raise Exception(model.signature_def)
    output = load_cached_output(cache_path)
    if output is None:
        if mode == "dicom":
            output = inference(tf_dataset, model)
            cache_output(output, cache_path)
        elif mode == "bids":
            output = test(monai_dataset, tf_dataset, model)
        else:
            raise Exception("Unknown mode")
    
    return output
    


# def main():
#     parser = argparse.ArgumentParser(prog="DeepDicomSort_MONAI", description='Bidscoin regex finder for DICOM datasets')
#     parser.add_argument("root_dir", type=str, help="Root directory of the dataset")
#     parser.add_argument("--label_indicator", dest="label_indicator", default="ProtocolName", type=str, help="Label indicator for the dataset, default is 'ProtocolName'")
#     args = parser.parse_args()
#     root_dir = args.root_dir
#     label_indicator = args.label_indicator
    
#     run(root_dir, label_indicator)

# main()