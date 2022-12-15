import cv2
import numpy as np


class TemplateMatcher:
    """
    OpenCV feature/template matcher
    """

    def __init__(self, descriptor: str = "SIFT", matcher_type: str = "BFM"):
        """
        Args:
            descriptor (str): The type of descriptor. Possible types are SIFT and ORB.
            matcher_type (str): The type of the matcher. Possible types are Brute-Force and FLANN.
                See https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        """
        self.descriptor = descriptor
        self.matcher_type = matcher_type

        if descriptor == "SIFT":
            self.detector = cv2.SIFT_create()
        elif descriptor == "ORB":
            self.detector = cv2.ORB_create()
        else:
            raise RuntimeError(f"Got an unknown descriptor: {descriptor}")

        if matcher_type == "BFM":
            norm_type = cv2.NORM_L2
            cross_check = False
            if descriptor == "ORB":
                norm_type = cv2.NORM_HAMMING
                cross_check = True
            self.matcher = cv2.BFMatcher(norm_type, cross_check)
        elif matcher_type == "FLANN":
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise RuntimeError(f"Got an unknown matcher: {matcher_type}")

    def _match(self, query_image: np.ndarray, reference_image: np.ndarray):
        """
        Run feature matcher on two images and store keypoints matching.
        Args:
            query_image (numpy.ndarray): A grayscale(1-channel) image to be queried.
            reference_image (numpy.ndarray): A grayscale(1-channel) template image.
        """
        kp1, des1 = self.detector.detectAndCompute(query_image, None)
        kp2, des2 = self.detector.detectAndCompute(reference_image, None)
        if self.descriptor != "ORB":
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good.append([m])
            matches = good
        else:
            matches = self.matcher.match(des1, des2)
            matches = [[m] for m in sorted(matches, key=lambda x: x.distance)]
        self.kp1 = kp1
        self.kp2 = kp2
        self.matches = matches

    def findHomography(self, query_image: np.ndarray, reference_image: np.ndarray):
        """
        Finds a perspective transformation between two image planes
        Args:
            query_image (numpy.ndarray): A grayscale(1-channel) image to be queried.
            reference_image (numpy.ndarray): A grayscale(1-channel) template image.
        """
        self._match(query_image, reference_image)
        if len(self.matches) > 20:
            src_pts = np.float32(
                [self.kp1[m[0].queryIdx].pt for m in self.matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [self.kp2[m[0].trainIdx].pt for m in self.matches]
            ).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # M is now the transformation matrix
            # from reference image to query image
            self.M = np.linalg.inv(M)
        else:
            raise RuntimeError("There are not enough of good matches")

    def getTransformationMat(self):
        """
        Returns a 3x3 M(numpy.ndarray) the transformation matrix calculated
        """
        assert self.M is not None
        return self.M

    def drawMatches(self, query_image: np.ndarray, reference_image: np.ndarray):
        """
        Draws matching keypoints on images and return the final image plot
        Args:
            query_image (numpy.ndarray): A grayscale(1-channel) image to be queried.
            reference_image (numpy.ndarray): A grayscale(1-channel) template image.
        """
        return cv2.drawMatchesKnn(
            query_image,
            self.kp1,
            reference_image,
            self.kp2,
            self.matches,
            None,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

    def transformImage(self, reference_image: np.ndarray):
        """
        Returns perspectively transformed reference image using the transformation matrix M(numpy.ndarray)
        Args:
            reference_image (numpy.ndarray): A grayscale(1-channel) template image used while finding the homography.
        """
        h, w = reference_image.shape[:2]
        return cv2.warpPerspective(reference_image, self.getTransformationMat(), (w, h))

    def overlayReferenceOnQuery(
        self, query_image: np.ndarray, reference_image: np.ndarray
    ):
        """
        Draws ground truths on the reference image, transforms and overlays it on the query image.
        Args:
            query_image (numpy.ndarray): A grayscale(1-channel) image to be queried.
            reference_image (numpy.ndarray): A grayscale(1-channel) template image used.
        """
        transformed_reference = self.transformImage(reference_image)
        return cv2.addWeighted(query_image, 0.5, transformed_reference, 0.5, 0.0)
