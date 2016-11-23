import os
import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filter import rank
from skimage import io
from skimage import exposure
from skimage import data
import matplotlib
import matplotlib.pyplot as plt
import dicom
import pylab

matplotlib.rcParams['font.size'] = 9


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[img.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    print("..................... img cdf")
    print(img_cdf)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf

def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    print(nda.shape)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()

# Directory where the DICOM files are being stored (in this
# case the 'MyHead' folder).
pathDicom = "/home/elliotnam/project/mamography/pilot_images/test/"

dFile =  dicom.read_file(pathDicom + "000135.dcm",)

print(dFile.pixel_array)

pylab.imshow(dFile.pixel_array,cmap=pylab.gray()) # pylab readings and conversion
pylab._imsave("tt.jpg",cmap=pylab.gray(),type="jpg")
pylab.show() #Dispaly
pylab.savefig(pathDicom+"tt.jpg")
# Z slice of the DICOM files to process. In the interest of
# simplicity, segmentation will be limited to a single 2D
# image but all processes are entirely applicable to the 3D image
idxSlice = 0
# int labels to assign to the segmented white and gray matter.
# These need to be different integers but their values themselves
# don't matter
labelWhiteMatter = 1
labelGrayMatter = 2


#reader = SimpleITK.ImageSeriesReader()
#filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
#print(filenamesDICOM)
##reader.SetFileNames(filenamesDICOM)
#reader.SetOutputPixelType(pixelID=SimpleITK.sitkInt16)
#imgOriginal = reader.Execute()

#imgOriginal = imgOriginal[:,:,idxSlice]

#print(imgOriginal)
#sitk_show(imgOriginal)
#img = SimpleITK.GetArrayFromImage(imgOriginal)

#print("............... size............")
#print(type(img))
#print(img.shape)
#print(img.size)
#img = img_as_ubyte(data.moon())
#print(img)
#mu, sigma = 100, 15
#hist,bin_cen = exposure.histogram(img)
print("hist.......................")
#np.savetxt("test.tt",hist,fmt="%s")
#print("bin_cen")
#print(bin_cen.shape)
#print(np.sum(hist))
#print(bin_cen)
#bin_cen = np.arange(256)

#print(type(bin_cen))

#print(hist.shape)
#print(hist)
#print(bin_cen)

plt.figure(figsize=(9, 4))
plt.subplot(131)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(132)
#plt.imshow(img , cmap='gray', interpolation='nearest')
plt.axis('off')
plt.subplot(133)
print(".......hist.shape............")
print(hist.shape)
print(bin_cen.shape)
plt.plot(bin_cen, hist, lw=2)
#plt.axvline(val, color='k', ls='--')

plt.tight_layout()
plt.show()
