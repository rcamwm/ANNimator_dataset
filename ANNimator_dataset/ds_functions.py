from . import pickle
from . import DataLoader, random_split

def save_pickle(dataset, path):
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)

def get_dataset(dataset_path, train_percent=1):
    with open(dataset_path, 'rb') as file:
        dataset = pickle.load(file)

    dataset_size = len(dataset)
    train_size = int(train_percent * dataset_size) 
    test_size = dataset_size - train_size
    return random_split(dataset, [train_size, test_size])
   
def get_data_loaders(dataset_path, batchsize=1, train_percent=1):
    train_dataset, test_dataset = get_dataset(dataset_path, train_percent)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    return train_loader, test_loader 