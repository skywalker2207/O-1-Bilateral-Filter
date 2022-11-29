# Fast O(1) Bilateral Filter using trigonometric range kernels

## Brief Description

We know that the edge-preserving bilateral filter involves an additional range kernel along with the spatial filter. This is used to restrict the averaging to those neighborhood pixels whose intensity are similar or close to that of the pixel of interest. The range kernel operates by acting on the pixel intensities. This makes the averaging process nonlinear and computationally intensive, particularly when the spatial filter is large. In this paper, we show how the Bilateral Filter can be implemented in O(1) time complexity using trigonometric range kernels. 

## Setup and Installation

- Requirements have been mentioned in `requirements.txt` 
- python environment of version 3.8 or higher is necessary for some inbuilt commands so the testing environment should be appropriate

## Sample outputs

- for same image and different kernel size the computational time of bilateral filter is nearly same but if the image size increases the computational time also increases
- it depends directy on the number of pixels in the image
- it has a time complexity of h*w(log(h*w))where h and w are the height and width of the image respectively

Time Analysis of the Fast Bilateral Filter
<img src="/data/outs/performance.png" width="300" height="200">

## Running the collaboratory notebook

- src folder contains the jupyter(.ipynb) notebook for the implementation of the bilateral filter using trigonometric range kernels
- all the computations , inputs and outputs are present in the notebook itself.
- while downloading zip file of the repository, make sure to extract the data folder in the same directory as the src folder(where out Project_Team_14.ipynb is present, it is the main file).
- run the notebook in the same order as the cells are present in the notebook.

## Team Information

### Team Name : Full Marks

#### Team Members : 
- Mitul Garg (2020102026)
- Atharv Sujlegaonkar (2020102025)
- Nikhil Agarwal (2020102021)
- Sreeharsha TD (2020102040)

## Biblography:
- Fast O(1) Bilateral Filter using trigonometric range kernels , Kunal Narayan Chaudhury, Student Member, IEEE, Daniel Sage, and Michael Unser, Fellow, IEEE , [link](https://github.com/Digital-Image-Processing-IIITH/dip-m22-project-full-marks/blob/main/docs/given_in_sheet%20(1).pdf) 

