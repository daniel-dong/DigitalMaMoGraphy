import dicom
import Image
import mudicom

#meta=dicom.read_file("dicomimage.dcm")
#TT=Image.frombuffer("L",imSize,meta.PixelData,"raw","",0,1)
#TT.save("testOUTPUT.tiff","TIFF",compression="none")

#mu = mudicom.load("/home/elliotnam/project/mamography/pilot_images/test/000135.dcm")

#img = mu.image
#print(img.numpy)

#img.save_as_pl("/home/elliotnam/project/mamography/pilot_images/test/000135.jpg")



from vtk import vtkDICOMImageReader
from vtk import vtkImageShiftScale
from vtk import vtkPNGWriter

reader = vtkDICOMImageReader()
reader.SetFileName('/home/elliotnam/project/mamography/pilot_images/test/000135.dcm')
reader.Update()
image = reader.GetOutput()

shiftScaleFilter = vtkImageShiftScale()
shiftScaleFilter.SetOutputScalarTypeToUnsignedChar()
shiftScaleFilter.SetInputConnection(reader.GetOutputPort())

shiftScaleFilter.SetShift(-1.0*image.GetScalarRange()[0])
oldRange = image.GetScalarRange()[1] - image.GetScalarRange()[0]
newRange = 255

shiftScaleFilter.SetScale(newRange/oldRange)
shiftScaleFilter.Update()

writer = vtkPNGWriter()
writer.SetFileName('output.jpg')
writer.SetInputConnection(shiftScaleFilter.GetOutputPort())
writer.Write()