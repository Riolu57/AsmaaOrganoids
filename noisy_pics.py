from os import walk
from os.path import join
from numpy.random import default_rng
from numpy import asarray, array, zeros, mgrid, dstack, vectorize, flip
from PIL import Image
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sn


class GaussianPics:
    def __init__(self, var: array, size: tuple, path: str, key: int=12345, figsize=(10, 5)):
        """Generates pictures of certain size with gaussian noise of given deviation.

        Images will be resized to the correct size if they differ.

        Usage:
            for img in GaussianPics(): ...

        Or:
            with GaussianPics() as pics:
                for pic in pics: ...

        :param var: The variance matrix of the gaussian distribution. Must be of shape 2x2.
        :param size: The size of the images. If needed, the images will be padded or compressed to the correct size.
        :param path: The path of the folder in which the images can be found.
        :param key: The random seed used to initialize the random number generator.
        """
        self.generator = default_rng(key)
        self.var = var
        self.size = size
        self.path = path
        self.figsize = figsize

    def __iter__(self):
        for path, _, img_paths in walk(self.path):
            for img_path in img_paths:
                mask = self.__generate_mask()
                img = Image.open(join(path, img_path))
                img_arr = asarray(img.resize(self.size))
                yield img_arr + mask

    def __get_mean(self):
        mean_range = asarray(range(0, self.size[0]))
        if self.size[0] == self.size[1]:
            mean = self.generator.choice(mean_range, size=(2,), replace=True)
        else:
            mean = (self.generator.choice(mean_range), self.generator.choice(mean_range))

        return mean

    def __generate_mask(self):
        mean = self.__get_mean()
        pos = dstack(mgrid[0:self.size[0]:1, 0:self.size[1]:1])
        # mask = self.__generate_mask_values(mean, pos)

        mask = zeros(self.size)
        mean_range = asarray(range(0, self.size[0]))
        mvn = multivariate_normal(mean, self.var)

        for x in mean_range:
            for y in mean_range:
                norm_mean = self.generator.choice([-1, 1])*mvn.pdf((x, y))
                # mask[x, y] = self.generator.normal(norm_mean, scale=0.1)
                mask[x, y] = self.generator.choice([-1, 1])*mvn.pdf((x, y))

        return mask

    # def __generate_mask_values(self, /, *, mean, loc):
    #     mvn = multivariate_normal(mean, self.var)
    #     norm_mean = self.generator.choice([-1, 1]) * mvn.pdf(loc)
    #     return self.generator.normal(norm_mean, scale=1)
    #
    # __generate_mask_values = vectorize(__generate_mask_values, cache=True)

    def vis_mv(self, save_path, granularity=0.1):
        mean = self.__get_mean()
        x, y = mgrid[0:self.size[0]:granularity, 0:self.size[1]:granularity]
        pos = dstack((x, y))
        rv = multivariate_normal(mean, self.var)
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        ax.contourf(x, y, rv.pdf(pos))
        fig.gca().invert_yaxis()
        # plt.title(f"Multivariate Normal with mean = ({mean[0]}, {mean[1]}), std = ({self.var[0, 0]}, {self.var[0, 1]} \\ {self.var[1, 0]}, {self.var[1, 1]}) ")

        fig.savefig(save_path)

    def vis_mask(self, save_path):
        data = self.__generate_mask()
        data = flip(data, 1)
        plt.figure(figsize=self.figsize)
        heat_map = sn.heatmap(data, annot=False)
        plt.savefig(save_path)
