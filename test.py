from data.BaseDataset import load_features_and_labels_task1


FEATURE_DIR = "/home/orisu/avi2026/dataset/autodl-tmp/train_feature"
BASE_DIR = "/home/orisu/avi2026/dataset/train_data"
LABEL_FILE = "/home/orisu/avi2026/dataset/train_data.csv"


data, labels = load_features_and_labels_task1(BASE_DIR,FEATURE_DIR, LABEL_FILE)

for key in data['q3'][0]:
    print(f"Question: {key} - shape: {data['q3'][0][key].shape}")
 