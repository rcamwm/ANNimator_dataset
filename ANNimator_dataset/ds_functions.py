from . import pickle
from . import DataLoader, random_split

def save_pickle(dataset, path):
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)

def get_dataset(dataset_path):
    with open(dataset_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset
   
def get_data_loaders(dataset_path, batchsize=1, train_percent=1):
    dataset = get_dataset(dataset_path)
    dataset_size = len(dataset)
    train_size = int(train_percent * dataset_size) 
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    return train_loader, test_loader 