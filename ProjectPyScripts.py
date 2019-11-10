import glob
import json
import os
from random import randint

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# df lists
dx_ints = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "nv": 4, "vasc": 5, "mel": 6}
dx_list = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]

ham_dir = "/home/marios/Downloads/skin-cancer-mnist-ham10000/ham10000_images"


def create_lookup_dict():
    """ creates dict for easier image label lookup """
    metadata = pd.read_csv("HAM10000_metadata.csv")
    metadata["dx_int"] = metadata["dx"].map(dx_ints)
    lookup_dict = metadata.loc[:, ["image_id", "dx_int"]].set_index("image_id")["dx_int"].to_dict()
    with open("lookup_dict.json", "w+") as f:
        json.dump(lookup_dict, f, indent=4)
    count_dict = metadata.groupby(["dx"]).size().reset_index().rename(columns={0: 'count'}).set_index("dx")[
        "count"].to_dict()
    with open("count_dict.json", "w+") as g:
        json.dump(count_dict, g, indent=4)


def get_lookup_dict():
    with open("lookup_dict.json", "r") as f:
        return json.load(f)


def get_count_dict():
    with open("count_dict.json", "r") as f:
        return json.load(f)


def crop_square(img_array):
    left_crop = randint(20, 130)
    right_crop = 150 - left_crop
    return img_array[:, left_crop:-right_crop]


def resize(img_array, side=28):
    return cv2.resize(img_array, (side, side), interpolation=cv2.INTER_AREA)


def squish(img_array):
    return cv2.resize(img_array, (450, 450), interpolation=cv2.INTER_AREA)


def grayscale(img_array):
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)


def crop_all_to_squares():
    imgs = glob.glob("/home/marios/Downloads/skin-cancer-mnist-ham10000/ham10000_images_part_2/*.jpg")

    metadata = pd.read_csv("HAM10000_metadata.csv")

    # print(os.path.join(os.getcwd(), im_name + ext))
    # im = cv2.imread(imgs[2223])

    # resize all
    # crop all
    # then run all rotations
    # repeat until required number per cat is reached

    imgs_flat_array = []

    for im in imgs:
        # read image
        im_array = cv2.imread(im)

        # create transformed image file path
        im_filename = os.path.basename(im)
        # im_name, ext = os.path.splitext(im_filename)
        # filepath = os.path.join(os.getcwd(), "augmented", "".join([im_name, "crop", ext]))
        #
        # cv2.imwrite(filepath, resize(crop_square(grayscale(im_array))))
        imgs_flat_array.append(np.ravel(resize(crop_square(grayscale(im_array)))))

    df = pd.DataFrame(imgs_flat_array)
    print(df.shape)
    df.to_csv("extracted.csv", index=False)


def process_and_augment_dataset():
    metadata = pd.read_csv("HAM10000_metadata.csv")
    count_dict = get_count_dict()
    lookup_dict = get_lookup_dict()
    target_no_imgs_per_dx = 3000

    # augment each dx separately
    for dx in dx_list[:2]:
        img_count = 0
        pass_count = 0

        # filter data for current dx
        metadata_filtered = metadata[metadata["dx"] == dx]

        # if dx count is larger that target, keep that count
        if count_dict[dx] > target_no_imgs_per_dx:
            total = count_dict[dx]
        else:
            total = target_no_imgs_per_dx

        with open("edited.csv", "a+") as f:
            # set up progress bar
            t = tqdm(total=total)
            t.set_description(f"Transforming {dx} images")

            # repeat loop until desired number of imgs is reached
            while img_count < total:
                # update progress bar
                if pass_count == 0:
                    t.set_postfix_str("Squishing images")
                else:
                    t.set_postfix_str(f"Randomly cropping/rotating/flipping images ({pass_count}x)")

                # shuffle df
                metadata_filtered_shuffled = metadata_filtered.sample(frac=1)

                # iterate over imgs and transform
                for index, row in metadata_filtered_shuffled.iterrows():
                    # stop if desired number is reached
                    if img_count >= total:
                        break

                    img_id = row["image_id"]
                    assert row["dx"] == dx

                    # read images
                    img_path = os.path.join(ham_dir, f"{img_id}.jpg")
                    img_array = cv2.imread(img_path)

                    # convert to grayscale
                    img_array = grayscale(img_array)

                    # keep record of the transformations applied
                    applied_transf = []

                    # apply transformations
                    if pass_count == 0:
                        # first pass: squish all
                        img_array = squish(img_array)
                        applied_transf.append("sq")
                    else:
                        # second and subsequent passes: first crop square
                        img_array = crop_square(img_array)
                        applied_transf.append("cr")

                        # randomly rotate in multiples of 90 (or not)
                        rotate = randint(0, 3)
                        img_array = np.rot90(img_array, rotate)
                        applied_transf.append(f"r{rotate}")

                        # flip/mirror (or not)
                        flip = randint(0, 1)
                        if flip == 1:
                            img_array = np.fliplr(img_array)
                        applied_transf.append(f"f{flip}")

                    # resize and flatten array, for appending to csv
                    img_array = np.ravel(resize(img_array))

                    # append img_id, dx as int, and applied transformations
                    img_array = np.append(img_array, [img_id])
                    img_array = np.append(img_array, [dx_ints[dx]])
                    img_array = np.append(img_array, ["-".join(applied_transf)])

                    # append to final to file
                    f.write(",".join(img_array))
                    f.write("\n")
                    del img_array
                    img_count += 1
                    t.update()

                pass_count += 1

            t.close()


def check_all_same_resolution():
    # imgs = glob.glob("test_images/*.jpg")
    imgs = glob.glob("/home/marios/Downloads/skin-cancer-mnist-ham10000/ham10000_images_part_2/*.jpg")
    sizes = any(sum(cv2.imread(im).shape) != 1053 for im in imgs)
    print(sizes)
    # im = cv2.imread("test_images/ISIC_0024313.jpg")
    # print(im.shape)
    # cv2.imshow("lesion", im)

    # shape = (im.shape[1], im.shape[0])
    #
    # matrix = cv2.getRotationMatrix2D(center=(450 / 2, 600 / 2), angle=90, scale=1)
    # image = cv2.warpAffine(src=im, M=matrix, dsize=shape)
    # cv2.imwrite('ISIC_0024313R.jpg', image)


def gallery(array, ncols=10):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def import_images():
    images = glob.glob("test_images/*.jpg")
    im_array = np.array([np.asarray(Image.open(im).convert("RGB")) for im in images])
    return im_array


def main():
    # array = import_images()
    # result = gallery(array)
    # plt.imshow(result)
    # plt.savefig("image_grid.pdf")
    # plt.close()
    # check_all_same_resolution()
    # print(cv2.getBuildInformation())
    # crop_all_to_squares()
    # cv2.imwrite("squished.jpg", squish(cv2.imread("test_images/ISIC_0024306.jpg")))
    # create_lookup_dict()
    process_and_augment_dataset()


if __name__ == '__main__':
    main()
