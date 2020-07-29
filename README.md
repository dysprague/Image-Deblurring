# Image-Deblurring
Encoder decoder deep neural network for the NTIRE 2020 image deblurring challenge.

Work done along with Thomas Huck and Bhrij Patel under the guidance of Cynthia Rudin and Rachel Draelos

The NTIRE 2020 Image Deblurring challenge presented competitors with a series of blurry camera videos.
The challenge was to produce a model that could reliably take the blurry camera videos and output
a deblurred version of the video. We created a encoder-decoder neural network that takes as input 
a set of 5 image segments (the frame to deblur as well as the surrounding frames) and outputs a single 
deblurred image. The segments are then pieced back together at the end to reproduce the full deblurred
frame. The model takes advantage of the RGB information by splitting the streams and learning each color
frame separately. It also relies on separable convolutions to improve the speed of computation without 
loss of power. We were fairly limited by our minimal access to GPU computing time, but we were still
able to produce a working model that could successfully deblur images.
