import torch
import Preprocessing
import SimpleITK as sitk
import vtk
import numpy as np
import os

## load templates
templatePath = "Templates/"
templateHeadMask = sitk.ReadImage(os.path.join(templatePath, 'templateHeadMaskRigid.mha'))
segmentationTemplate = sitk.ReadImage(os.path.join(templatePath, 'templateSegmentation.mha'))

reader = vtk.vtkPolyDataReader()
reader.SetFileName(os.path.join(templatePath, 'landmarks_glabella.vtk'))
reader.Update()
templateLandmarks = reader.GetOutput()

## load ct and landmarks
ctImage = sitk.ReadImage('CTImage.mha')

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName('CranialBaseLandmarks.vtp')
reader.Update()
landmarks = reader.GetOutput()

## prepare model input
processed = Preprocessing.ProcessInstance(ctImage, landmarks, templateHeadMask, segmentationTemplate, templateLandmarks)

device = 'cuda:0'
ageScale = 5.
SexScale = 2.5
image = torch.tensor(np.expand_dims(np.expand_dims(processed,axis=0),axis=0), device = device)

## set up target demographics
age = 0.5  ## target age -- between 0-1, 1 represents 10 years of age
sex = 1. ## 0 or 1, 1 for female


## transform age and sex to tensor
age = ((torch.tensor(age, device=device).float() * 200 + 1).log() * ageScale).view(-1,1) ## log transform age
sex = (torch.tensor(sex, device=device) * SexScale).view(-1,1)
y = torch.cat([age, sex], dim = 1)

### load jit model
Encoder = torch.jit.load('Encoder.pth').to(device)
Encoder.eval()
Decoder = torch.jit.load('Decoder.pth').to(device)
Decoder.eval()

encoded_feature = Encoder(image)
prediction = Decoder(encoded_feature, y)

prediction = sitk.GetImageFromArray(prediction.detach().cpu().numpy()[0,0])
cnnImageSize = np.array([128, 128, 128], dtype=np.int16)
prediction.SetOrigin(segmentationTemplate.GetOrigin())
prediction.SetSpacing(np.array(segmentationTemplate.GetSpacing()) * np.array(segmentationTemplate.GetSize())/ cnnImageSize)

sitk.WriteImage(prediction, 'prediction.mha')
print("Prediction successful!\n\tAge: ", age, "\n\tSex: ", sex, "\n\tSynthesized CT saved to prediction.mha")
