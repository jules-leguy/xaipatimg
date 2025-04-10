from torchvision.transforms import transforms

resnet18_preprocess_no_norm = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])