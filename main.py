from noisy_pics import GaussianPics
from numpy import asarray
from PIL import Image

path = r"U:\Year 4\Asmaa\Data\images_to_David\Stacks"
size = (200, 200)
var = asarray([[100, 30], [30, 100]])

pics = GaussianPics(var=var, size=size, path=path)
pics.vis_mv("./example_mvn.png")
pics.vis_mask("./example_mask.png")

# for pic in pics:
#     img = Image.fromarray(pic)
#     img.show()
