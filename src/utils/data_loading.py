import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import os
import torch.nn.functional as func
from filelock import FileLock
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, TensorDataset


class DatasetType:
    """
    Enum to determine dataset type.
    """
    train = 0
    eval = 1
    test = 2


# MNIST Variation
def mnist_variation_loader(type, batch_size=256, num_workers=0, dims=1, pin_memory=False):
    if dims != 1:
        print("MNIST Variations loader has no support for dims.")

    train_valid_filepath = os.path.relpath(
        "../data/mnist_rotation_back_image_new/mnist_all_background_images_rotation_normalized_train_valid.amat")
    test_filepath = os.path.relpath(
        "../data/mnist_rotation_back_image_new/mnist_all_background_images_rotation_normalized_test.amat")

    if type == DatasetType.train or type == DatasetType.eval:

        # Reading train/valid data
        train_valid_data = []
        with open(train_valid_filepath) as f:
            line = f.readline()
            curr_data = []
            line_count = 1
            while line != "":
                cur_line = line

                while cur_line != "\n":
                    partition = cur_line.partition(" ")
                    if partition[0] == "":
                        if partition[2] == "":
                            break
                        else:
                            cur_line = partition[2]
                            continue
                    curr_data.append(float(partition[0]))
                    cur_line = partition[2]

                train_valid_data.append(curr_data)
                curr_data = []
                line = f.readline()
                line_count += 1

        train_valid_data = np.array(train_valid_data)

        train_valid_targets = train_valid_data[:, -1]
        train_valid_targets = train_valid_targets.astype('int')

        train_valid_data = train_valid_data[:, :784]

        if type == DatasetType.train:
            data = train_valid_data[:-2000, :]
            targets = train_valid_targets[:-2000]
        else:
            data = train_valid_data[-2000:, :]
            targets = train_valid_targets[-2000:]

    else:
        # Reading test data
        test_data = []
        with open(test_filepath) as f:
            line = f.readline()
            curr_data = []
            line_count = 1
            while line != "":
                cur_line = line

                while cur_line != "\n":
                    partition = cur_line.partition(" ")
                    if partition[0] == "":
                        if partition[2] == "":
                            break
                        else:
                            cur_line = partition[2]
                            continue
                    curr_data.append(float(partition[0]))
                    cur_line = partition[2]

                test_data.append(curr_data)
                curr_data = []
                line = f.readline()
                line_count += 1

        test_data = np.array(test_data)
        targets = test_data[:, -1].astype('int')
        data = test_data[:, :784]

    data = torch.FloatTensor(data)
    targets = torch.Tensor(targets)
    targets = func.one_hot(targets.to(torch.int64)).to(torch.float32)

    dataset = TensorDataset(data, targets)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


# MNIST
def mnist_loader(type, batch_size=256, num_workers=0, dims=1, pin_memory=False):
    def one_hot_mnist(targets):
        targets_onehot = torch.zeros(10)
        targets_onehot[targets] = 1
        return targets_onehot

    dataset = datasets.MNIST

    if dims == 3:
        transform_all = transforms.Compose([
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform_all = transforms.Compose([
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda a: a.view(-1))
        ])

    is_train = (True if type == DatasetType.train else False)

    root = os.path.join(os.path.dirname(__file__), "../../data/")
    assert os.path.isdir(root)

    lock = os.path.join(os.path.dirname(__file__), "../../data/MNIST.lock")
    with FileLock(lock):
        dataset = dataset(root=root, train=is_train, download=True, transform=transform_all,
                          target_transform=one_hot_mnist)

        if is_train:
            sampler = RandomSampler(dataset)

        else:
            print(len(dataset))
            index = int(len(dataset) * 0.2) if (type == DatasetType.eval) else int(len(dataset) * 0.8)
            indices = list(range(index)) if (type == DatasetType.eval) else list(range(index, len(dataset)))
            sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)

        return loader

# CIFAR
def cifar10_loader(type, batch_size=256, num_workers=0, dims=1, pin_memory=False):
    def one_hot_cifar(targets):
        targets_onehot = torch.zeros(10)
        targets_onehot[targets] = 1
        return targets_onehot

    dataset = datasets.CIFAR10

    if dims == 3:
        transform_all = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform_all = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda a: a.view(-1))
        ])

    is_train = (True if type == DatasetType.train else False)

    root = os.path.join(os.path.dirname(__file__), "../../data/")
    assert os.path.isdir(root)

    lock = os.path.join(os.path.dirname(__file__), "../../data/CIFAR10.lock")
    with FileLock(lock):
        dataset = dataset(root=root, train=is_train, download=True, transform=transform_all,
                          target_transform=one_hot_cifar)

        if is_train:
            sampler = RandomSampler(dataset)

        else:
            index = int(len(dataset) * 0.2) if (type == DatasetType.eval) else int(len(dataset) * 0.8)
            indices = list(range(index)) if (type == DatasetType.eval) else list(range(index, len(dataset)))
            sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)

        return loader


def cifar100_loader(type, batch_size=256, num_workers=0, dims=1, pin_memory=False):
    def one_hot_cifar(targets):
        targets_onehot = torch.zeros(100)
        targets_onehot[targets] = 1
        return targets_onehot

    dataset = datasets.CIFAR100

    if dims == 3:
        transform_all = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform_all = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda a: a.view(-1))
        ])

    is_train = (True if type == DatasetType.train else False)

    root = os.path.join(os.path.dirname(__file__), "../../data/")
    assert os.path.isdir(root)

    lock = os.path.join(os.path.dirname(__file__), "../../data/CIFAR10.lock")
    with FileLock(lock):
        dataset = dataset(root=root, train=is_train, download=True, transform=transform_all,
                          target_transform=one_hot_cifar)

        if is_train:
            sampler = RandomSampler(dataset)

        else:
            index = int(len(dataset) * 0.2) if (type == DatasetType.eval) else int(len(dataset) * 0.8)
            indices = list(range(index)) if (type == DatasetType.eval) else list(range(index, len(dataset)))
            sampler = SubsetRandomSampler(indices)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)

        return loader


# BANANA_CAR
def banana_car_loader(dataset_type, size=(280, 190), batch_size=256, num_workers=0, pin_memory=False):
    def one_hot_bc(targets):
        targets_onehot = torch.zeros(2)
        targets_onehot[targets] = 1
        return targets_onehot

    transform_all = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.view(-1))
    ])

    root = os.path.join(os.path.dirname(__file__), "../../data/banana_car")
    assert os.path.isdir(root)

    dataset = datasets.ImageFolder(root=root, transform=transform_all, target_transform=one_hot_bc)

    indices = np.array(list(range(len(dataset))))
    np.random.shuffle(indices)

    if dataset_type == DatasetType.train:
        indices = indices[:int(len(dataset) * 0.7)]

    elif dataset_type == DatasetType.eval:
        indices = indices[int(len(dataset) * 0.7):int(len(dataset) * 0.8)]

    elif dataset_type == DatasetType.test:
        indices = indices[int(len(dataset) * 0.8):]

    else:
        raise ReferenceError

    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader


def bananacar_abstract_loader(size=(280, 190), batch_size=256, num_workers=0, pin_memory=False):
    """Loader for the images containing cars with banana shapes"""

    def one_hot_one_one(targets):
        return torch.FloatTensor([1, 1])

    transform_all = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda a: a.view(-1))
    ])

    path = "../data/abstraction_eval_bananacar"
    assert os.path.isdir(path)

    dataset = datasets.ImageFolder(root=path, transform=transform_all,
                                   target_transform=one_hot_one_one)

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader

# Equations
def equations_loader(batch_size=256, num_workers=0, pin_memory=False):
    eqs = [
        lambda inputs: 1.0 if np.sum(inputs) < (len(inputs) * 0.25) else 0.0,
        lambda inputs: 1.0 if (len(inputs) * 0.5) > np.sum(inputs) > (len(inputs) * 0.25) else 0.0,
        lambda inputs: 1.0 if (len(inputs) * 0.75) > np.sum(inputs) > (len(inputs) * 0.5) else 0.0,
        lambda inputs: 1.0 if np.sum(inputs) > (len(inputs) * 0.75) else 0.0
    ]

    inputs = np.random.rand(10000, 10).astype('f')
    targets = np.asarray([[eq(i) for eq in eqs] for i in inputs]).astype('f')

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader
