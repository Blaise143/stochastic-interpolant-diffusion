import torchvision


def emnist_transform():
    return torchvision.transforms.Compose([
        transforms.Lambda(lambda x: transforms.functional.rotate(x, -90)),
        transforms.Lambda(lambda x: transforms.functional.hflip(x)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
