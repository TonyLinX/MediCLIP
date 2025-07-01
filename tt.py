from datasets.dataset import FabricTestDataset
from torchvision import transforms
from easydict import EasyDict

dataset = FabricTestDataset(
    args=EasyDict({'image_size': 224}),
    source="data/fabric",
    preprocess=transforms.ToTensor()
)

print("Start loading first 5 samples...")
for i in range(5):
    sample = dataset[i]
    print(f"[Sample {i}] class = {sample['classname']}, is_anomaly = {sample['is_anomaly']}")
