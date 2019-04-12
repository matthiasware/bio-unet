General domain description
==========================

Mitotic count is an important parameter in breast cancer grading as it gives an evaluation of the agressiveness of the 
tumour. However, mitosis detection is a challenging problem and has not been addressed well in the literature. This is due
to the fact that mitosis are:
	- small objects
	- with a large variety of shape configurations.
The four main phases of a mitosis are:
	- prophase
	- metaphase
	- anaphase and
	- telophase.
The shape of the nucleus is very different depending on the phase of the mitosis. On its last stage, the telophase, a mitosis has two distinct nuclei, but they are not yet full individual cells. 

The data you receive are H&E stained histological images of different breast cancers prepared on 5 
slides.

Our objective is to develop a deep learning based semantic segmentation algorithm that support histophathologists in the detecting of
mitosis on these type of images.

Data description
================

The data is organized as follows.

Each scanned image consists of a raw image, the desired output
as a pixel list and an overlay of both.

For example:

Input image: A00_00.png
Annotations: A00_00.csv
Overlay: A00_00.jpg

The annotations are organized one mitosis per line.
Each line lists the coordinates of all pixels belonging
to one mitosis segment.

Solution to prepare
====================

1) Split the data into a training and a test dataset
2) Implement using python and tensorflow a basic solution for mitotic cell segmentation using the U-Net architecture or a similar alternative (U-Net paper: https://arxiv.org/pdf/1505.04597.pdf)%E5%92%8C%5bTiramisu%5d(https://arxiv.org/abs/1611.09326.pdf)
3) Evaluate your solution on the test dataset

Questions to prepare
====================
1) What are the advantages and shortcomings of this approach?
2) Would there be feasible alternatives?
3) How would you further improve the method?
