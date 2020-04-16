import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import math



class Filtering:
    image = None
    filter_name = None
    filter_size = None
    alpha_d = None
    order = None

    def __init__(self, image, filter_name, filter_size, alpha_d=2, order = 1.5 ):
        """initializes the variables of spatial filtering on an input image
        takes as input:
        image: the noisy input image
        filter_name: the name of the mask to use
        filter_size: integer value of the size of the mask
        alpha_d: parameter of the alpha trimmed mean filter
        order: parameter of the order for contra harmonic"""

        self.image = image
        if filter_name == 'median':
            self.filter = self.get_median_filter
        elif filter_name == 'min':
            self.filter = self.get_min_filter
        elif filter_name == 'max':
            self.filter = self.get_max_filter
        elif filter_name == 'alpha_trimmed':
            self.filter = self.get_alpha_filter
        elif filter_name == 'arithmetic_mean':
            self.filter = self.get_arithmetic_mean_filter
        elif filter_name == 'geometric_mean':
            self.filter = self.get_geo_mean_filter
        elif filter_name == 'contra_harmonic':
            self.filter = self.get_contra_harmonic_filter

        self.filter_size = filter_size
        self.alpha_d = alpha_d
        self.order = order


    def get_median_filter(self, kernel):
        """Computes the median filter
        takes as input:
        kernel: a list/array of intensity values
        returns the median value in the current kernel
        """
        image = cv2.imread('Lenna.png')
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(image,-1,kernel)

        return dst

    def get_min_filter(self, kernel,image,L):
        """Computes the minimum filter
        takes as input:
        ikernel: a list/array of intensity values
        returns the minimum value in the current kernel"""
        image = np.asarray(image, dtype=np.float)
        image = image*kernel + L
        # clip pixel values
        image[image > 255] = 255
        image[image] 

        return image

    def get_max_filter(self, kerne,image):
        """Computes the maximum filter
        takes as input:
        kernel: a list/array of intensity values
        returns the maximum value in the current kernel"""
        (B,G,R)=cv2.split(image)
        M=np.maximum(np.maximum(R,G),B)
        R[R < M]=0
        G[G < M]=0
        B[B < M]=0
        return cv2.merge([B,G,R])

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#             help="path to input image")
# args = vars(ap.parse_args())
#         # load the image, apply the max RGB filter, and show the
#         # output images
# image = cv2.imread(args["image"])
# filtered = max_rgb_filter(image)
# cv2.imshow("Images", np.hstack([image, filtered]))
# cv2.waitKey(0)

    def get_alpha_filter(self,kernal,alpha_d,frame_1,frame_2,mask):

        """Computes the median filter
        takes as input:
        kernel: a list/array of intensity values
        alpha_d: clip off parameter for the alpha trimmed filter
        returns the alpha trimmed mean value in the current kernel"""
        alpha_d=mask/225.0
        blended=cv2.convertScaleAbs(frame_1*(1-alpha_d)+frame_2*alpha_d)
        return blended

    def get_arithmetic_mean_filter(self, kernel):
        """Computes the arithmetic mean filter
        takes as input:
        kernel: a list/array of intensity values
        returns the arithmetic mean value in the current kernel"""
        kernel= np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
        sharpened_img = cv2.filter2D(sp_05, -1, kernel_sharpening)           
        return sharpened_img

    def get_geo_mean_filter(self, kernel):
        """Computes the geometric mean filter
                        takes as input:
                        kernel: a list/array of intensity values
                        returns the geometric mean value in the current kernel"""
        image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE).astype(float)
        rows, cols = image.shape[:2]
        kernel = 5
        padsize = int((kernel-1)/2)
        pad_img = cv2.copyMakeBorder(image, *[padsize]*4, cv2.BORDER_DEFAULT)
        geomean1 = np.zeros_like(image)
        for r in range(rows):
            for c in range(cols):
                geomean1[r, c] = np.prod(pad_img[r:r+kernel, c:c+kernel])**(1/(kernel**2))
        geomean1 = np.uint8(geomean1)
        return geomean1

# cv2.imshow('1', geomean1)
# cv2.waitKey()
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)


    def get_contra_harmonic_filter(self, kernel, order):
        """Computes the harmonic filter
                        takes as input:
        kernel: a list/array of intensity values
        order: order paramter for the
        returns the harmonic mean value in the current kernel"""
        image = cv2.imread(lenna.png)[:, :, 0]
        homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
        img_filtered = homo_filter.filter(I=image, filter_params=[30,2])
        return img_filtered
# cv2.imwrite(img_path_out, img_filtered)



    def filtering(self, I, filter_params, filter='butterworth', H = None):
        """performs filtering on an image containing gaussian or salt & pepper noise
        returns the denoised image
        ----------------------------------------------------------
        Note: Here when we perform filtering we are not doing convolution.
        For every pixel in the image, we select a neighborhood of values defined by the kernal and apply a mathematical
        operation for all the elements with in the kernel. For example, mean, median and etc.

        Steps:
        1. add the necesssary zero padding to the noisy image, that way we have sufficient values to perform the operati
        ons on the pixels at the image corners. The number of rows and columns of zero padding is defined by the kernel size
        2. Iterate through the image and every pixel (i,j) gather the neighbors defined by the kernel into a list (or any data structure)
        3. Pass these values to one of the filters that will compute the necessary mathematical operations (mean, median, etc.)
        4. Save the results at (i,j) in the ouput image.
        5. return the output image
        """
        if len(I.shape) is not 2:

            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter=='butterworth':
            H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='gaussian':
            H = self.__gaussian_filter(I_shape = I_fft.shape, filter_params = filter_params)
        elif filter=='external':
            print('external')
            if len(H.shape) is not 2:
                raise Exception('Invalid external filter')
        else:
            raise Exception('Selected filter not implemented')
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)


        # return self.image
