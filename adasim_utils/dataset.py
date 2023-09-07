from torchvision.datasets.folder import default_loader, DatasetFolder, IMG_EXTENSIONS, ImageFolder
from typing import Any, Callable, Optional
import torch
from collections import defaultdict
from collections import OrderedDict
import torch.nn.functional as FF


class DatasetFolderAdaSim(DatasetFolder):
    def __init__(
            self,
            root: str,
            args,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            return_index_instead_of_target: Optional[bool] = False):
        self.args = args
        self.return_index_instead_of_target = return_index_instead_of_target
        self.root = root
        classes, _ = self.find_classes(self.root)
        super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.nn_matrix_cpu = None
        self.sim_matrix_cpu = None

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.nn_matrix_cpu is not None:
            # ~ 50 x 10 (vote_nn x topk)
            current_sim = self.sim_matrix_cpu[index]
            current_nn = self.nn_matrix_cpu[index]

            # Get sum of similarities for each index
            dict_lst = [defaultdict(float, zip(current_nn[n].tolist(), current_sim[n].tolist())) for n in
                        range(self.args.vote_nn_nb)]
            output_dict = OrderedDict({k: sum(d[k] for d in dict_lst) for k in
                                       set([item for sublist in current_nn.tolist() for item in sublist])})
            sorted_items = sorted(output_dict.items(), key=lambda item: item[1], reverse=True)[:self.args.topk]
            keys, items = list(zip(*sorted_items))

            if keys[0] != index:
                sample2 = sample
            else:
                # Sample from distribution arising from the topk biggest sums
                sampling_distribution = FF.softmax(
                    torch.Tensor(items) / self.args.vote_nn_nb / self.args.sampling_softmax_temp, dim=-1)
                sampled_relative_index = sampling_distribution.multinomial(num_samples=1).item()
                sampled_absolute_index = keys[sampled_relative_index]
                path2, target2 = self.samples[sampled_absolute_index]
                sample2 = self.loader(path2)
        else:
            sample2 = sample

        same_im = 1 if sample == sample2 else 0

        if self.transform is not None:
            sample = self.transform(sample, sample2)

        # TODO: targets are the ones from sample (not sample2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index_instead_of_target:
            target = index

        return sample, target, same_im


class ImageNetReturnPath(ImageFolder):

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
