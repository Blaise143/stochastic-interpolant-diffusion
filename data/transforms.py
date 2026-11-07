import torchvision


def rotate_90(img):
    """Rotates an image -90 degrees.

    torchvision's raw EMNIST images are stored rotated/transposed relative
    to how digits are normally displayed; this undoes that.

    Args:
        img: Image to rotate.
    """
    return torchvision.transforms.functional.rotate(img, -90)


def hflip(img):
    """Horizontally flips an image.

    Combined with `rotate_90`, this corrects the orientation quirk in the
    raw EMNIST dataset.

    Args:
        img: Image to flip.
    """
    return torchvision.transforms.functional.hflip(img)


def emnist_transform():
    """Builds the standard preprocessing pipeline for EMNIST images.

    Fixes EMNIST's orientation, converts to tensor, and normalizes to [-1, 1].

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    return torchvision.transforms.Compose([
        rotate_90,
        hflip,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
