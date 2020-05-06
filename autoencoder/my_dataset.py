import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class myDataset(Dataset):

	def __init__(self, csv_file):
		"""
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self. = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform




    # def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    #     self.root = os.path.expanduser(root)
    #     self.transform = transform
    #     self.target_transform = target_transform
    #     self.train = train  # training set or test set

    #     if download:
    #         self.download()

    #     if not self._check_exists():
    #         raise RuntimeError('Dataset not found.' +
    #                            ' You can use download=True to download it')

    #     if self.train:
    #         self.train_data, self.train_labels = torch.load(
    #             os.path.join(self.root, self.processed_folder, self.training_file))
    #     else:
    #         self.test_data, self.test_labels = torch.load(
    #             os.path.join(self.root, self.processed_folder, self.test_file))

    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (image, target) where target is index of the target class.
    #     """
    #     if self.train:
    #         img, target = self.train_data[index], self.train_labels[index]
    #     else:
    #         img, target = self.test_data[index], self.test_labels[index]

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target

    # def __len__(self):
    #     if self.train:
    #         return len(self.train_data)
    #     else:
    #         return len(self.test_data)



# class Dataset(data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, list_IDs, labels):
#         'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         X = torch.load('data/' + ID + '.pt')
#         y = self.labels[ID]

#         return X, y