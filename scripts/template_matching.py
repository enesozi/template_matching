import argparse

from feature_matcher.template_matcher import TemplateMatcher
from feature_matcher.utils import visualize_image, read_color_and_gray


def main(args):
    descriptor = "SIFT"
    matcher_type = "BFM"
    template_matcher = TemplateMatcher(descriptor=descriptor, matcher_type=matcher_type)
    query_img_bgr, query_img_gray = read_color_and_gray(args.query_image_path)
    ref_img_bgr, ref_img_gray = read_color_and_gray(args.reference_image_path)
    template_matcher.findHomography(
        query_image=query_img_gray, reference_image=ref_img_gray
    )
    debug_img = template_matcher.drawMatches(
        query_image=query_img_gray, reference_image=ref_img_gray
    )
    visualize_image(image=debug_img)
    ref_on_query = template_matcher.overlayReferenceOnQuery(
        query_image=query_img_bgr, reference_image=ref_img_bgr
    )
    visualize_image(image=ref_on_query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Template Matcher using OpenCV")
    parser.add_argument(
        "--query_image_path",
        help="The path to the query image",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--reference_image_path",
        help="The path to the reference image",
        required=True,
    )
    args = parser.parse_args()
    main(args)
