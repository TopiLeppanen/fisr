# fisr
Implementation of Fast Image Super-resolution algorithm

Implements the algorithm in the paper:
Fast Image Super-resolution Based on In-place Example Regression,
by Jianchao Yang, Zhe Lin, Scott Cohen (2013)

The algorithm needs training data set. For example The Berkeley Segmentation Dataset (500 images) can be used, or any other set of natural images.

This doesn't completely work yet. It doesn't seem to learn the sharpening filters properly.
This implementation doesn't implement the "Selective Patch Processing" as described in the paper. Implementing it would have a chance of just speeding up the upscaling part of the algorithm.
