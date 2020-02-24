import torch
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

class FakeData(VisionDataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the datset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10,
                 transform=None, target_transform=None, random_offset=0):
        super(FakeData, self).__init__(None, transform=transform,
                                       target_transform=target_transform)
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

        #rng_state = torch.get_rng_state()
        torch.manual_seed(self.random_offset)
        self.data = torch.rand(self.size, *self.image_size)
        self.targets = torch.randint(self.num_classes, size=(self.size,), dtype=torch.long)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        
        #torch.set_rng_state(rng_state)
        img, target = self.data[index], self.targets[index]
        
        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.size