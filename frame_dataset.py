import os
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class FrameDataset(Dataset):
    def __init__(self, folder_name, key_filenames, between_filenames):
        self.folder_name = folder_name
        self.key_filenames = key_filenames
        self.between_filenames = between_filenames
        
    def __len__(self):
        return len(self.key_filenames)
    
    def __getitem__(self, index):
        to_tensor = transforms.ToTensor()
        key_frame_1 = to_tensor(Image.open(os.path.join(self.folder_name, self.key_filenames[index][0])))
        key_frame_2 = to_tensor(Image.open(os.path.join(self.folder_name, self.key_filenames[index][1])))
        between_frame = to_tensor(Image.open(os.path.join(self.folder_name, self.between_filenames[index])))
        return key_frame_1, key_frame_2, between_frame

    @staticmethod
    def save_pickle(dataset, path):
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)
    
    @staticmethod
    def get_data_loaders(dataset_path, batchsize=1, train_percent=1):
        with open(dataset_path, 'rb') as file:
            dataset = pickle.load(file)

        dataset_size = len(dataset)
        train_size = int(train_percent * dataset_size) 
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        return train_loader, test_loader
