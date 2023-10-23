from . import os
from . import transforms
from . import Dataset
from . import Image

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