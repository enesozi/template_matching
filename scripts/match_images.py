import argparse
import cv2
import numpy as np
from pathlib import Path
from feature_matcher.template_matcher import TemplateMatcher
from feature_matcher.utils import (
    RoI,
    choose_keypoints,
    read_color_and_gray,
)


def main(args):
    descriptor = "SIFT"
    matcher_type = "BFM"
    template_matcher = TemplateMatcher(descriptor=descriptor, matcher_type=matcher_type)
    query_img_bgr, _ = read_color_and_gray(args.query_image_path)
    ref_img_bgr, _ = read_color_and_gray(args.reference_image_path)

    image = cv2.hconcat((query_img_bgr, ref_img_bgr))
    clone = image.copy()
    roi = RoI(image=image, clone_image=clone)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", choose_keypoints, roi)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", roi.image)
        key = cv2.waitKey(1) & 0xFF

        # press 'r' to reset the window
        if key == ord("r"):
            roi.image = clone.copy()

        # if the 'q' key is pressed, break from the loop
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()

    query_points = np.array(roi.points[0::2])
    ref_points = np.array(roi.points[1::2])
    # The reference image was shifted to the right
    ref_points[:, 0] -= query_img_bgr.shape[1]
    M, _ = cv2.findHomography(query_points, ref_points, cv2.RANSAC, 5.0)
    # Save transformation matrix
    with open(f"data/{Path(args.query_image_path).stem}.npy", "wb") as f:
        np.save(f, M)
    template_matcher.M = np.linalg.inv(M)
    img = template_matcher.overlayReferenceOnQuery(
        query_image=query_img_bgr, reference_image=ref_img_bgr
    )
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    key = cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Template Matching Manually")
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
