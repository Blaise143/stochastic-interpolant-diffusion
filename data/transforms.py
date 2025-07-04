import torchvision


def rotate_90(img):
    return torchvision.transforms.functional.rotate(img, -90)


def hflip(img):
    return torchvision.transforms.functional.hflip(img)


def emnist_transform():
    return torchvision.transforms.Compose([
        rotate_90,
        hflip,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
