import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

"""
Class for getting the values for padding a tensor image to square dimensions.
The call function allows transforms.compose to call SquarePad.
"""
class SquarePad:
    def __init__(self, return_padding=False):
        self.padding = 0, 0, 0, 0
        self.return_padding = return_padding

    def __call__(self, tensor_image):
        return self.pad_to_square(tensor_image)

    """
    Calculates square padding dimensions of tensor image,
    returns (side1, side2) plus/minus 0.5 for uneven padding
    """
    def prep_pad(self, width, height):
        # No padding if dims are equal
        if width == height:
            return 0, 0

        # Difference of dims
        pad = abs(width - height) / 2

        # safety check that difference is not 1
        if pad == 1:
            return 1, 1

        # Safety check for uneven for float after division
        if int(pad) != pad:
            return int(pad + .5), int(pad - .5)
        else:
            return int(pad), int(pad)

    """
    Pads a tensor image to a square (constant padding),
    returns the padded tensor image
    """
    def pad_to_square(self, tensor_image):
        height, width = tensor_image.size()[1], tensor_image.size()[2]
        side1, side2 = self.prep_pad(width, height)
        if width == height:
            padding = (0, 0, 0, 0)
        elif width > height:
            padding = (0, 0, side1, side2) # tuple reads left, right, top, bottom
        else:
            padding = (side1, side2, 0, 0)

        self.padding = padding
        transformed = F.pad(tensor_image, padding, mode='constant', value = 0)
        if self.return_padding:
            return transformed, self.padding
        else:
            return transformed

    """ Removes the padding from an image given the original and the padded img
    """
    @staticmethod
    def remove_padding(tensor_image, padded_image):
        size = tensor_image.size()
        return TF.center_crop(padded_image, (size[1], size[2]))
