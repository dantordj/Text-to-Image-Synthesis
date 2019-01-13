import torch
from models.gan_factory import gan_factory
import matplotlib.pyplot as plt
from txt2image_dataset import Text2ImageDataset
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from PIL import Image, ImageColor, ImagePalette
import matplotlib.cm as cm

cuda = torch.cuda.is_available()
def inception_score(images, inception_model, batch_size, resize=True, splits=5, generated_images=False):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = batch_size

    assert batch_size > 0
    assert N >= batch_size

    # Set up dtype

    if cuda:

        dtype = torch.cuda.FloatTensor
    else:

        dtype = torch.FloatTensor

    # Load inception model


    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    batch = images.type(dtype)
    batchv = Variable(batch)
    i = 0
    preds_i = get_pred(batchv)
    batch_size_i = preds_i.shape[0]

    preds[i*batch_size:(i*batch_size + batch_size_i)] = preds_i
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def quantizetopalette(silf, palette, dither=False):
    """Convert an RGB or L mode image to use a given P image's palette."""

    silf.load()

    # use palette from reference image
    palette.load()

    if palette.mode != "P":
        raise ValueError("bad mode for palette image")
    if silf.mode != "RGB" and silf.mode != "L":
        raise ValueError(
            "only RGB or L mode images can be quantized to a palette"
            )
    im = silf.im.convert("P", 1 if dither else 0, palette.im)
    # the 0 above means turn OFF dithering

    # Later versions of Pillow (4.x) rename _makeself to _new
    try:
        return silf._new(im)
    except AttributeError:
        return silf._makeself(im)



def colour_score(imgs, txts, threshold=20):
    colors_ref = ["blue", "red", "white", "cyan", "orange", "yellow", "brown",  "green", "pink", "black", "purple", "gray"]
    #colors = ["blue"]
    rgb_colors = {ImageColor.getrgb(c):c for c in colors_ref}
    #print(rgb_colors)
    palette = []
    step = 256 // (len(colors_ref) - 1)
    comp = 256 % (len(colors_ref) - 1)
    scores = []
    out_scores = []
    list_probs = []
    list_probs_cond = []
    for i, rgb_c in enumerate(rgb_colors.keys()):
        if i == 0:
            step_size = comp
        else:
            step_size = step
        palette += [rgb_c[0], rgb_c[1], rgb_c[2]] * step_size

    for color in colors_ref:
        color = "red"
        probs_cond = 0
        probs_not_cond = 0
        for i, image in enumerate(imgs):

            img = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
            width, height = img.size   # Get dimensions
            left = width/4
            top = height/4
            right = 3 * width/4
            bottom = 3 * height/4
            img = img.crop((left, top, right, bottom))
            palimage = Image.new('P', (16, 16))
            #palette_img = ImagePalette.ImagePalette(palette=palette, mode="P", size=len(palette))
            palimage.putpalette(palette)
            #palimage.load()
            #newimage = img.quantize(colors=len(colors), palette=palimage)
            #palimage.putpalette(palette*64)
            newimage = quantizetopalette(img, palimage, dither=False)
            """
            if i == 0:
                img.show()
                newimage.show()
            """
            newimage.putalpha(0)
            colors_img = newimage.getcolors(newimage.size[0] * newimage.size[1])

            colors_img_rgb = [rgb_colors[t[1][:3]] for t in colors_img if t[0] >= threshold]

            txt = txts[i]
            txt = txt.split(" ")

            colors_txt = set(colors_ref).intersection(txt)
            colors_common = set(colors_txt).intersection(colors_img_rgb)
            if len(colors_txt) == 0:
                continue
            scores += [len(colors_common) / len(colors_txt)]
            if color in colors_img_rgb:
                probs_not_cond += 1 / len(imgs)
            if color in colors_common:
                probs_cond += 1 / len(imgs)
            #out_scores += [len()]
        if probs_not_cond != 0:
            probs_cond /= probs_not_cond
            list_probs_cond += [probs_cond]
            list_probs += [probs_not_cond]

    list_probs_cond = np.array(list_probs_cond)
    list_probs = np.array(list_probs)

    return np.mean(list_probs_cond / list_probs), np.std(list_probs_cond / list_probs)
        #print(colors)
        #img = Image.fromarray(arr_img, mode="RGB")
        # img.putalpha(0)
