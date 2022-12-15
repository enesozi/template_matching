import cv2
import numpy as np
import matplotlib.path as mplPath
from dataclasses import dataclass, field


@dataclass
class RoI:
    image: np.ndarray
    clone_image: np.ndarray
    points: list = field(default_factory=list)
    drawing: bool = False

    def reset(self):
        self.drawing = False
        self.image = self.clone_image.copy()


@dataclass
class PolyLineRoI(RoI):
    prev_x: int = -1
    prev_y: int = -1
    start_x: int = -1
    start_y: int = -1


def read_color_and_gray(image_path):
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_gray


def visualize_image(image):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    key = cv2.waitKey(-1)
    cv2.destroyAllWindows()


def shape_selection(event, x, y, flags, params: RoI):

    # if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        params.points = [(x, y)]
        params.drawing = True
    # mouse is being moved, draw rectangle
    elif event == cv2.EVENT_MOUSEMOVE:
        if params.drawing:
            cv2.rectangle(params.image, params.points[0], (x, y), (0, 255, 0), -1)
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        params.reset()
        # draw a rectangle around the region of interest
        cv2.rectangle(params.image, params.points[0], (x,y), (0, 255, 0), 5)


# mouse callback function
def draw_lines(event, x, y, flags, params: PolyLineRoI):

    # if the left mouse button was clicked, record the starting
    if event == cv2.EVENT_LBUTTONDOWN:

        # draw circle of 4px
        cv2.circle(params.image, (x, y), 4, (0, 0, 127), -1)
        if params.prev_x != -1:
            cv2.line(
                params.image,
                (params.prev_x, params.prev_y),
                (x, y),
                (0, 0, 127),
                2,
                cv2.LINE_AA,
            )
            params.points[-1].append([x, y])
        else:  # if ix and iy are first points, store as starting points
            params.start_x, params.start_y = x, y
            params.points.append([[x, y]])
        params.prev_x, params.prev_y = x, y

    if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN:
        params.prev_x, params.prev_y = -1, -1
        cv2.line(
            params.image, (x, y), (params.start_x, params.start_y), (0, 0, 127), 2, cv2.LINE_AA
        )


def choose_keypoints(event, x, y, flags, params: RoI):
    # if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw circle of 10px
        cv2.circle(params.image, (x, y), 10, (0, 0, 127), -1)
        params.points.append([x, y])


def filter_keypoints_inside_the_polygon(
    keypoints: list[cv2.KeyPoint], polygon: mplPath.Path
):
    return [kp for kp in keypoints if polygon.contains_point(kp.pt)]


def draw_keypoints(keypoints: list[cv2.KeyPoint], image: np.ndarray):
    for kp in keypoints:
        pt = list(map(int, kp.pt))
        cv2.circle(image, center=pt, radius=1, color=(0, 0, 255), thickness=2)
