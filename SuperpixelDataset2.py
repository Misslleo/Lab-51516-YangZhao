import torch
from torch_geometric.data import InMemoryDataset
from joblib import Parallel, delayed
from tqdm import tqdm
from superpixel.t2 import FlameSet
import dill as pickle

val_dataset = FlameSet('C:/Users/Yang Zhao/PycharmProjects/rob/outputt/val')

class SuperpixelSCDataset2(InMemoryDataset):

    def __init__(self, root,
                 dataset_name,
                 superpix_size: int,
                 edgeflow_type,
                 processor_type,
                 image_processor,
                 dataset_size: int,
                 n_jobs: int = 16):
        self.dataset = dataset_name
        self.n_jobs = n_jobs
        self.dataset_size = dataset_size
        self.dataset_name = dataset_name
        self.name = f"{dataset_name}_{superpix_size}_{edgeflow_type.__name__}_{processor_type.__class__.__name__}"
        folder = f"{root}/{self.name}"

        self.processor_type = processor_type
        self.ImageProcessor = image_processor(superpix_size, edgeflow_type)

        self.pre_transform = lambda image: processor_type.process(self.ImageProcessor.image_to_features(image))

        super(SuperpixelSCDataset2, self).__init__(root, pre_transform=self.pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices["X0"]) - 1

    def load_dataset(self):
        """Load the dataset_processor from here and process it if it doesn't exist"""
        print("Loading dataset_processor from disk...")
        data, slices = torch.load(self.processed_paths[0])
        return data, slices

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return ["features.pt"]

    def process(self):
        print(f"Pre-transforming {self.dataset_name} dataset..")
        data_list = Parallel(n_jobs=self.n_jobs, prefer="threads") \
            (delayed(self.pre_transform)(image) for image in tqdm(val_dataset))

        print(f"Finished pre-transforming {self.train_str} dataset.")
        data, slices = self.processor_type.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        return self.processor_type.get(self.data, self.slices, idx)

    def get_name(self):
        return self.name