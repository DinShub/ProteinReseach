from datasets import load_dataset

data_files = {"train": "train.csv", "test": "test.csv"}
datasets = load_dataset('csv', data_files=data_files, delimiter='\t')
print(datasets)