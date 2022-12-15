import argparse
import numpy as np
from pathlib import Path

from feature_matcher.template_matcher import TemplateMatcher
from feature_matcher.utils import read_color_and_gray, visualize_image


def main(args):
    template_matcher = TemplateMatcher()
    data_folder = Path(args.data_folder).resolve()
    image_list = Path.glob(data_folder,pattern="*.jpg")
    ref_img_path = Path(args.reference_image).resolve()
    ref_img = read_color_and_gray(str(ref_img_path))[0]
    for img_p in image_list:
        if img_p.stem == ref_img_path.stem:
            continue
        tf_mat = np.load(img_p.with_suffix(".npy"))
        img = read_color_and_gray(str(img_p))[0]
        template_matcher.M = tf_mat
        ref_img = template_matcher.overlayReferenceOnQuery(query_image=ref_img,reference_image=img)
    visualize_image(ref_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Template Matching Manually")
    parser.add_argument(
        "--data_folder",
        help="The path to the query images to be transformed onto the reference. Make sure transformation matrices are also in the same directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--reference_image",
        help="The path to the reference image",
        required=True,
        type=str
    )
    args = parser.parse_args()
    main(args)
