import argparse
import pandas as pd
from src.utils.all_utils import create_directory, read_yaml
import os
import joblib
from sklearn.linear_model import ElasticNet

def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    train_data_path = config["artifacts"]["train"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    train_data_file_path = os.path.join(artifacts_dir,split_data_dir, train_data_path)
    train_df = pd.read_csv(train_data_file_path)
    # print(train_df.head())
    train_y = train_df["quality"]
    train_x = train_df.drop("quality", axis=1)

    alpha = params["model_params"]["ElasticNet"]["alpha"]
    l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]
    random_state = params["base"]["random_state"]
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)

    model_dir = config["artifacts"]["model_dir"]
    model_filename = config["artifacts"]["model_filename"]

    model_dir = os.path.join(artifacts_dir, model_dir)

    create_directory([model_dir])

    model_path = os.path.join(model_dir, model_filename)


    joblib.dump(lr, model_path)


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c",default="config/config.yaml")
    args.add_argument("--params","-p",default="params.yaml")
    parsed_args = args.parse_args()
    train(parsed_args.config, parsed_args.params)