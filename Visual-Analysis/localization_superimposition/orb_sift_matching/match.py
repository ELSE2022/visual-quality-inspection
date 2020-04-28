import cv2
import numpy as np
import os

from image_info import Image
from utils import multiDimenDist

################################################################################

level = 8
number = 100

################################################################################


class Match:

    def __init__(self, IM, IM1, _matches_number=10):
        self.Image = IM
        self.Image1 = IM1
        self.new_mask = self.Image.mask.copy()
        self.axe0 = 0
        self.axe1 = 0

        self.matches_number = _matches_number
        self.point_distance = 20
        self.proportion = 1

        self.sift_created = False
        self._build_features('orb')
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self._match()
        self.shift = self._control_features(0, 0.02)
        print('Best features selected are: ', self.shift)
        print(' Images are scaled {0} times in respect to '
        	'each other.'.format(self.proportion))
        self.matches = self.matches[:self.matches_number]
        # TODO add factory for image comparison

    def __repr__(self):
        return '<Match object ({} {})>'.format(self.IM.image_name,
                                               self.matches_number)

    def _match(self):
    	# Feature matching based on BF selected.
    	# controls the number of the matches.
        self.matches = self.bf.match(self.des1, self.des2)
        if len(self.matches) < self.matches_number:
            self.matches_number = len(self.matches)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def get_mask(self):
        # TODO: scale the mask
        self.new_mask = np.roll(self.new_mask, int(self.get_coords(
            self.shift[0])[1][1] - self.get_coords(self.shift[0])[0][1]),
                                axis=0)
        self.new_mask = np.roll(self.new_mask, int(self.get_coords(
            self.shift[0])[1][0] - self.get_coords(self.shift[0])[0][0]),
                                axis=1)
        self.axe0 = int(self.get_coords(
            self.shift[0])[1][1] - self.get_coords(self.shift[0])[0][1])
        self.axe1 = int(self.get_coords(
            self.shift[0])[1][0] - self.get_coords(self.shift[0])[0][0])

    def sift_matching(self):
    	# SIFT features matching algorithm
        self._build_features('sift')
        self.sift_created = True
        self.bf = cv2.BFMatcher()
        self.matches = self.bf.knnMatch(self.des1, self.des2, k=2)
        good = []
        for m,n in self.matches:
        	if m.distance < 0.9*n.distance:
        		good.append([m])
        good = [x[0] for x in good]
        self.matches = sorted(good, key = lambda x: x.distance)
        return self._control_features(0, 0.02)

    def _build_features(self, _features):
        if _features == 'sift':
            # SIFT initialization.
            self.build = cv2.xfeatures2d.SIFT_create(100, nOctaveLayers=8)
            print('SIFT initialized.')
        elif _features == 'orb':
            # ORB initialization.
            self.build = cv2.ORB_create()
            print('ORB initialized.')
        # computes the features per image.
        self.kp1, self.des1 = self.build.detectAndCompute(self.Image.image, None)
        self.kp2, self.des2 = self.build.detectAndCompute(self.Image1.image, None)

    def vis(self, framename='matching'):
        # type: (_ind) -> str
        try:
            framename = str(framename)
        except Exception:
            framename = 'matching'
            print('Wrong frame name format! The filename changed to '
                  '"matching.png"! ')
        img3 = cv2.drawMatches(self.Image.image, self.kp1, self.Image1.image,
                               self.kp2, self.matches[:self.matches_number],
                               None, flags=2)
        img3 = cv2.circle(img3, tuple(self.get_coords(self.shift[0])[0]), 3,
                          (0, 0, 255), -1)
        img3 = cv2.circle(img3, tuple(self.get_coords(self._second)[0]), 3,
                          (255, 0, 0), -1)
        img3 = cv2.circle(img3, tuple(self.get_coords(self.shift[1])[0]), 3,
                          (0, 255, 0), -1)
        #cv2.imwrite(framename + '.png', img3)
        #cv2.imwrite(framename + '_mask.png', self.new_mask)

    ###################
    # Utility functions

    def get_coords(self, _ind):
        # type: (_ind) -> int
        coords1 = self.kp1[self.matches[int(_ind)].queryIdx].pt
        coords2 = self.kp2[self.matches[int(_ind)].trainIdx].pt
        coords1 = [int(x) for x in coords1]
        coords2 = [int(x) for x in coords2]
        # print(coords1)
        return coords1, coords2

    ###################
    # Control functions

    def _get_proportions(self, _ind1, _ind2):
        # type: (_ind1) -> int
        # type: (_ind2) -> int
        try:
            return multiDimenDist(self.get_coords(int(_ind1))[0],
                                  self.get_coords(int(_ind2))[0]) / multiDimenDist(
                self.get_coords(int(_ind1))[1], self.get_coords(int(_ind2))[1])
        except Exception:
            return 100

    def _get_ideal_proportion(self, _init):
        for _i in range(_init + 1, self.matches_number):
            # check that initial points are far enough from each other
            if multiDimenDist(self.get_coords(int(_init))[0],
                              self.get_coords(int(_i))[0]) > self.point_distance:
                self._second = int(_i)
                return self._get_proportions(int(_init), int(_i)), int(_init)

        if _init < self.matches_number:
            # increases initial point for next search is all the points are too
            # close to the previous one
            return self._get_ideal_proportion(_init+1)
        self._second = 1
        return self._get_proportions(0, 1), 0, 1

    def _control_features(self, _init, _interval):
        # type: (_init) -> int
        # the concept of _ideal_proportion is based on the estimate that the
        # most similar features will be the most correct ones (doesn't always
        # prove right).
        _ideal_proportion = self._get_ideal_proportion(int(_init))[0]
        _init = self._get_ideal_proportion(int(_init))[1]

        for _i in range(_init + 2, self.matches_number):
            # check whether next point is far enough from the init point on both
            # the images (perfect and compared)
            if (multiDimenDist(self.get_coords(int(_init))[0],
                               self.get_coords(int(_i))[0]) >
                self.point_distance) and \
                    (multiDimenDist(self.get_coords(int(_init))[1],
                                    self.get_coords(int(_i))[1]) >
                     self.point_distance) and \
                    (multiDimenDist(self.get_coords(int(self._second))[0],
                                    self.get_coords(int(_i))[0]) >
                     self.point_distance):
                # check whether the proportion between the initial point and the
                # next point is similar enough to the best match proportion
                # called --_ideal_proportion--
                if (_ideal_proportion - _interval <= self._get_proportions(
                        _init, _i) <= _ideal_proportion + _interval) and \
                        (_ideal_proportion - _interval <= self._get_proportions(
                            self._second, _i) <= _ideal_proportion + _interval):
                    self.proportion = _ideal_proportion
                    return _init, _i
        if _init >= self.matches_number - 3:
            print("Could not find paired features in the images. Increasing "
                  "the distance between the features to {0}.".format(_interval + 0.02))
            if _interval > 0.2:
                if self.sift_created == False:
                    print("Could not find paired features with ORB. "
                          "Initializing SIFT features.")
                    return self.sift_matching()
                else:
                    print("Could not find paired features at all.")
                    return 0, 1
                # return 0, 1

            _interval = _interval + 0.02
            return self._control_features(0, _interval)

        return self._control_features(_init + 1, _interval)

