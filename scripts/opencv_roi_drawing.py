import argparse
import cv2

from feature_matcher.utils import (
    PolyLineRoI,
    draw_lines,
    shape_selection,
    choose_keypoints,
    read_color_and_gray,
)


def main(args):
    query_img_bgr, _ = read_color_and_gray(args.query_image_path)
    ref_img_bgr, _ = read_color_and_gray(args.reference_image_path)

    image = cv2.hconcat((query_img_bgr, ref_img_bgr))
    clone = image.copy()

    roi = PolyLineRoI(image=image, clone_image=clone)
    drawing_method = draw_lines
    if args.shape == "circle":
        drawing_method = choose_keypoints
    elif args.shape == "rectangle":
        drawing_method = shape_selection
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", drawing_method, roi)
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

    # close all open windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("OpenCV drawing exmaple")
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
    parser.add_argument(
        "--shape",
        help="Define the shape you want to draw on the image",
        choices=["circle", "rectangle", "polygon"],
        default="polygon",
    )
    args = parser.parse_args()
    main(args)
