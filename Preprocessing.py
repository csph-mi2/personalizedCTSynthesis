import skimage
import skimage.measure
import numpy as np
import SimpleITK as sitk
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import torchio as tio
import vtk
from scipy import spatial
from vtk.util import numpy_support

def CreateHeadMask(ctImage, hounsfieldThreshold = -200):
    """
    Returns a binary image mask of the head from an input CT image

    Parameters
    ----------
    ctImage: sitkImage
        A scalar CT image
    hounsfieldThreshold: int
        Hard threshold used to binarize the CT image

    Returns
    -------
    sitkImage
        A binary image of the head
    """

    headMask = sitk.GetArrayFromImage(ctImage)

    # Getting the head
    headMask = (headMask > hounsfieldThreshold).astype(np.uint8)

    headMask = skimage.measure.label(headMask)
    largestLabel = np.argmax(np.bincount(headMask.flat)[1:])+1
    headMask = (headMask == largestLabel).astype(np.uint8)

    headMask = sitk.GetImageFromArray(headMask)
    headMask.SetOrigin(ctImage.GetOrigin())
    headMask.SetSpacing(ctImage.GetSpacing())
    headMask.SetDirection(ctImage.GetDirection())

    return headMask

def CreateBoneMask(ctImage, headMaskImage=None, minimumThreshold=160, maximumThreshold=160, verbose=False, ):
    """
    Uses adapting thresholding to create a binary mask of the cranial bones from an input CT image.
    [Dangi et al., Robust head CT image registration pipeline for craniosynostosis skull correction surgery, Healthcare Technology Letters, 2017]

    Parameters
    ----------
    ctImage: sitkImage
        A scalar CT image
    headMaskImage: sitkImage
        A binary image of the head
    minimumThreshold: int
        The lower threshold of the range to use for adapting thresholding
    maximumThreshold: int
        The upper threshold of the range to use for adapting thresholding
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    sitkImage
        A binary image of the cranial bones
    """

    # If a head mask is not provided
    if headMaskImage is None:

        if verbose:
            print('Creating head mask.')

        headMaskImage = CreateHeadMask(ctImage)


    ctImageArray = sitk.GetArrayFromImage(ctImage)
    headMaskImageArray = sitk.GetArrayViewFromImage(headMaskImage)

    # Appling the mask to the CT image
    ctImageArray[headMaskImageArray == 0] = 0

    # Extracting the bones
    minObjects = np.inf
    optimalThreshold = 0
    for threshold in range(minimumThreshold, maximumThreshold+1, 10):

        if verbose:
            print('Optimizing skull segmentation. Threshold {:03d}.'.format(threshold), end='\r')

        labels = skimage.measure.label(ctImageArray >= threshold)
        nObjects = np.max(labels)

        if nObjects < minObjects:
            minObjects = nObjects
            optimalThreshold = threshold
    if verbose:
        print('The optimal threshold for skull segmentation is {:03d}.'.format(optimalThreshold))
    
    ctImageArray = ctImageArray >= optimalThreshold

    ctImageArray = skimage.measure.label(ctImageArray)
    largestLabel = np.argmax(np.bincount(ctImageArray.flat)[1:])+1
    ctImageArray = (ctImageArray == largestLabel).astype(np.uint)
    
    ctImageArray = sitk.GetImageFromArray(ctImageArray)
    ctImageArray.SetOrigin(ctImage.GetOrigin())
    ctImageArray.SetSpacing(ctImage.GetSpacing())
    ctImageArray.SetDirection(ctImage.GetDirection())

    return ctImageArray

def AlignLandmarksWithTemplate(landmarks, landmarks_target, scaling=False, verbose=False):

    """
    Calculates analytically the least-squares best-fit transform between corresponding 3D points A->B.

    Parameters
    ----------
    landmarks: vtkPolyData
        Cranial base landmarks
    scaling: bool
        Indicates if the calculated transformation is purely rigid (False) or contains scaling (True)
    useGlabella: bool
        if True, the landmarks are located at the glabella, temporal processes of the dorsum sellae and opisthion.
        If False, the landmarks are located at the nasion, temporal processes of the dorsum sellae and opisthion.
    verbose: bool
        Indicates if the function will print information in the standard output 

    Returns
    -------
    np.array
        Rotation (+scaling) matrix with shape 3x3
    np.array
        Translation vector with shape 3
    """

    # Reading template landmarks
    # reader = vtk.vtkPolyDataReader()
    # if useGlabella:
    #     reader.SetFileName(GLABELLA_CRANIALBASE_LANDMARKS_PATH)
    # else:
    #     reader.SetFileName(NASION_CRANIALBASE_LANDMARKS_PATH)
    # reader.Update()
    # templateLandmarks = reader.GetOutput()

    npLandmarks = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    npTemplateLandmarks = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):

        npLandmarks[p,:3] = np.array(landmarks.GetPoint(p))
        npTemplateLandmarks[p,:] = np.array(landmarks_target.GetPoint(p))

    
    R, t = RegisterPointClouds(npLandmarks, npTemplateLandmarks, scaling=scaling)

    center = np.mean(npLandmarks, axis=0).astype(np.float64)

    #transform = sitk.Similarity3DTransform()
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(R.ravel())
    transform.SetCenter(center)
    transform.SetTranslation(t)
    
    return transform

def RegisterPointClouds(A, B, scaling=False):
    """
    Calculates analytically the least-squares best-fit transform between corresponding 3D points A->B.
    Parameters
    ----------
    A: np.array
        Moving point cloud with shape Nx3, where N is the number of points
    B: np.array
        Fixed point cloud with shape Nx3, where N is the number of points
    scaling: bool
        Indicates if the calculated transformation is purely rigid (False) or contains isotropic scaling (True)

    Returns
    -------
    np.array
        Rotation (+scaling) matrix with shape 3x3
    np.array
        Translation vector with shape 3
    """

    assert len(A) == len(B) # Both point clouds must have the same number of points

    zz = np.zeros(shape=[A.shape[0],1])
    A = np.append(A, zz, axis=1)
    B = np.append(B, zz, axis=1)
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Scaling
    if scaling:
        AA = np.dot(R.T, AA.T).T
        s = np.mean(np.linalg.norm(BB[:,:3], axis=1)) / np.mean(np.linalg.norm(AA[:,:3], axis=1))
        #R *= s
    else:
        s = 1

    R = R[:3,:3]#.T
    t = (centroid_B - centroid_A)[:3]

    if scaling:
        return s, R, t,
    else:
        return R, t

def FindLandmarkTransformation(templateLandmarks, subjectLandmarks):

    # Converting to numpy
    npTemplateLandmarks = np.zeros([templateLandmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    npLandmarks = np.zeros([subjectLandmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(subjectLandmarks.GetNumberOfPoints()):
        npTemplateLandmarks[p,:] = np.array(templateLandmarks.GetPoint(p))
        npLandmarks[p,:3] = np.array(subjectLandmarks.GetPoint(p))

    # Registering
    s, R, t = RegisterPointClouds(npLandmarks, npTemplateLandmarks, scaling=True)

    center = np.mean(npLandmarks, axis=0).astype(np.float64)

    transform = sitk.Similarity3DTransform()
    transform.SetMatrix(R.ravel())
    transform.SetScale(s)
    transform.SetCenter(center)
    transform.SetTranslation(t)

    #transform = sitk.AffineTransform(3)
    #transform.SetMatrix((s * R).ravel())
    #transform.SetCenter(center)
    #transform.SetTranslation(t)

    return transform

def FloodFillHull(image):
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img

def CreateMeshFromBinaryImage(binaryImage, insidePixelValue=1):
    """
    Uses the marching cubes algorithm to create a surface model from a binary image

    Parameters
    ----------
    binaryImage: sitkImage
        The binary image
    insidePixelValue: {int, float}
        The pixel value to use for mesh creation

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """

    numpyImage = sitk.GetArrayViewFromImage(binaryImage).astype(np.ubyte)
    
    dataArray = numpy_support.numpy_to_vtk(num_array=numpyImage.ravel(),  deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)

    vtkImage = vtk.vtkImageData()
    vtkImage.SetSpacing(binaryImage.GetSpacing()[0], binaryImage.GetSpacing()[1], binaryImage.GetSpacing()[2])
    vtkImage.SetOrigin(binaryImage.GetOrigin()[0], binaryImage.GetOrigin()[1], binaryImage.GetOrigin()[2])
    vtkImage.SetExtent(0, numpyImage.shape[2]-1, 0, numpyImage.shape[1]-1, 0, numpyImage.shape[0]-1)
    vtkImage.GetPointData().SetScalars(dataArray)

    filter = vtk.vtkMarchingCubes()
    filter.SetInputData(vtkImage)
    filter.SetValue(0, insidePixelValue)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    return mesh

def RegisterImages(fixedImage, movingImage, initialTransform, fixedMask=None, movingMask=None, useSDFs=False, verbose=False):

    ## Calculating distance maps
    t = sitk.Similarity3DTransform()
    t.SetIdentity()

    if useSDFs:
        fixedImage = sitk.SignedDanielssonDistanceMapImageFilter().Execute(sitk.Cast(fixedImage, sitk.sitkUInt8))
        movingImage = sitk.SignedDanielssonDistanceMapImageFilter().Execute(sitk.Cast(movingImage, sitk.sitkUInt8))
    else:
        fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)
        movingImage = sitk.Cast(movingImage, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(initialTransform, inPlace=False)
    registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-6, numberOfIterations=1000, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000, costFunctionConvergenceFactor=1e+7)
    registration.SetShrinkFactorsPerLevel(shrinkFactors = [4, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[0.1, 0.1])
    registration.SetMetricSamplingPercentagePerLevel([0.5, 0.25])
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if fixedMask is not None:
        registration.SetMetricFixedMask(fixedMask)
    if movingMask is not None:
        registration.SetMetricMovingMask(movingMask)

    if verbose:
        registration.AddCommand(sitk.sitkIterationEvent,  
                            lambda:
                                print('Global alignment. Level: {}. Iteration: {:03d}. Metric: {:03.6f}.'.format(registration.GetCurrentLevel(), registration.GetOptimizerIteration() , registration.GetMetricValue()), end='\r')
                            )

    finalTransform = registration.Execute(fixedImage, movingImage)

    if verbose:
        print()

    return finalTransform

def CutMeshWithCranialBaseLandmarks(mesh, landmarks, extraSpace=0, useTwoLandmarks=False):
    """
    Crops the input surface model using the planes defined by the input landmarks

    Parameters
    ----------
    mesh: vtkPolyData
        Cranial surface model
    landmarks: vtkPolyData
        Cranial base landmarks (4 points)
    extraSpace: int
        Indicates the amount of extract space to keep under the planes defined by the cranial base landmarks
    useTwoLandmarks: bool
        Indicates if the cut is done only using the first and fourth landmarks, or using the two planes defined by all the landmarks

    Returns
    -------
    vtkPolyData
        The resulting surface model
    """


    landmarkCoords = np.zeros([landmarks.GetNumberOfPoints(), 3], dtype=np.float32)
    for p in range(landmarks.GetNumberOfPoints()):
        landmarkCoords[p,:] = np.array(landmarks.GetPoint(p))
    if not useTwoLandmarks:
        

        # normal of first plane
        v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
        v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
        n0 = np.cross(v0, v1)
        n0 = n0 / np.sqrt(np.sum(n0**2))
    
        ###########
        ## Moving landmark coordinates 1 cm away from cranial base so we don't miss the squamousal suture

        distanceToMove = (extraSpace/100.0) * np.abs(np.dot(np.mean(landmarkCoords[1:3,:], axis=0, keepdims=False) - landmarkCoords[3,:], n0))

        landmarkCoords[1:3,:] +=  (n0*distanceToMove).reshape((1,3))

        # Recalculating normal of first plane
        v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
        v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
        n0 = np.cross(v0, v1)
        n0 = n0 / np.sqrt(np.sum(n0**2))
        ###########
    
        # normal of second plane
        v0 = landmarkCoords[2, :] - landmarkCoords[3, :] # For plane 2 (posterior)
        v1 = landmarkCoords[1, :] - landmarkCoords[3, :]
        n1 = np.cross(v0, v1)
        n1 = n1 / np.sqrt(np.sum(n1**2))

        plane1 = vtk.vtkPlane()
        plane1.SetNormal(-n0)
        plane2 = vtk.vtkPlane()
        plane2.SetNormal(-n1)

        plane1.SetOrigin(landmarkCoords[0,:])
        plane2.SetOrigin(landmarkCoords[3,:])

        intersectionFunction = vtk.vtkImplicitBoolean()
        intersectionFunction.AddFunction(plane1)
        intersectionFunction.AddFunction(plane2)
        intersectionFunction.SetOperationTypeToIntersection()
    else:
        
        if extraSpace > 0:
            # normal of first plane
            v0 = landmarkCoords[1, :] - landmarkCoords[0, :] # For plane 1 (frontal)
            v1 = landmarkCoords[2, :] - landmarkCoords[0, :]
            n0 = np.cross(v0, v1)
            n0 = n0 / np.sqrt(np.sum(n0**2))
            landmarkCoords[0,:] +=  (n0*extraSpace)

            # normal of second plane
            v0 = landmarkCoords[3, :] - landmarkCoords[2, :] # For plane 1 (frontal)
            v1 = landmarkCoords[3, :] - landmarkCoords[1, :]
            n0 = np.cross(v0, v1)
            n0 = n0 / np.sqrt(np.sum(n0**2))
            landmarkCoords[3,:] +=  (n0*extraSpace)
        
        # Normal to plane
        dorsumVect = landmarkCoords[1, :] - landmarkCoords[2, :]
        dorsumVect = dorsumVect / np.sqrt(np.sum(dorsumVect**2))

        p0 = landmarkCoords[0, :] + 10 * dorsumVect
        p1 = landmarkCoords[0, :] - 10 * dorsumVect
        p2 = landmarkCoords[3, :]


        v0 = p2 - p1
        v1 = p2 - p0
        n = np.cross(v0, v1)
        n = n / np.sqrt(np.sum(n**2))


        plane = vtk.vtkPlane()
        plane.SetNormal(-n)
        plane.SetOrigin(p2)


        intersectionFunction = plane

    #cutter = vtk.vtkClipPolyData()
    cutter = vtk.vtkExtractPolyDataGeometry()
    cutter.ExtractInsideOff()
    cutter.SetInputData(mesh)
    #cutter.SetClipFunction(intersectionFunction)
    cutter.SetImplicitFunction(intersectionFunction)
    cutter.Update()

    return cutter.GetOutput()

def ProcessInstance(ctImage, landmarks, templateHeadMask, segmentationTemplate, templateLandmarks):
    
    cnnImageSize = np.array([128, 128, 128], dtype=np.int16)
    maxQuantile = 1947.0

    headMask = CreateHeadMask(ctImage, hounsfieldThreshold = -70)
    headMask = CreateBoneMask(ctImage, headMask, minimumThreshold=180, maximumThreshold=180)
    headMesh = CreateMeshFromBinaryImage(headMask)
    headMaskCut = CutMeshWithCranialBaseLandmarks(headMesh,landmarks)

    image = sitk.Image(ctImage.GetSize(), sitk.sitkUInt8)
    image.SetOrigin(ctImage.GetOrigin())
    image.SetSpacing(ctImage.GetSpacing())
    
    coords = np.array(headMaskCut.GetPoints().GetData())
    for i in range(coords.shape[0]):
        print(i, end='\r')
        idx = (coords[i] - image.GetOrigin()) / image.GetSpacing()
        image[int(np.round(idx[0])), int(np.round(idx[1])), int(np.round(idx[2]))] = 1
    
    convexMask = FloodFillHull(sitk.GetArrayFromImage(image))
    
    headMask = sitk.GetImageFromArray(convexMask)
    headMask.CopyInformation(ctImage)

    ctArray = sitk.GetArrayFromImage(ctImage)
    ctArray[convexMask==0] = 0
    ctArray[ctArray < 0] = 0
    ctImage_masked = sitk.GetImageFromArray(ctArray)
    ctImage_masked.CopyInformation(ctImage)

    # # toTemplateRigidTransform = sitk.ReadTransform(os.path.join(inputPatientPath, patientId, rigidTransformFileName))
    # toTemplateRigidTransform = AlignLandmarksWithTemplate(landmarks, templateLandmarks, scaling=False)

    # resampler = sitk.ResampleImageFilter()
    # resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetReferenceImage(segmentationTemplate)
    # resampler.SetOutputPixelType(sitk.sitkFloat32)
    # resampler.SetDefaultPixelValue(0)
    # resampler.SetTransform(toTemplateRigidTransform.GetInverse())
    # ctImageRigid = resampler.Execute(ctImage_masked)
    
    templateImageArray = np.zeros(np.flip(cnnImageSize), dtype=np.float32) # z, y, x
    templateImage = sitk.GetImageFromArray(templateImageArray)

    templateImage.SetOrigin(segmentationTemplate.GetOrigin())
    templateImage.SetSpacing(np.array(segmentationTemplate.GetSpacing()) * np.array(segmentationTemplate.GetSize())/ cnnImageSize)
    
    # # Downsample images
    # transform = sitk.AffineTransform(3)
    # transform.SetIdentity()
    # resampledCTImage = sitk.Resample(ctImageRigid, templateImage, transform.GetInverse(), sitk.sitkLinear)
    ##############################################
    ##############################################
    ##############################################

    #### fix by registering the whole head

    subjectToTemplateSimilarityTransform = FindLandmarkTransformation(templateLandmarks, landmarks)

    subjectToTemplateAffineTransform = sitk.Similarity3DTransform()
    subjectToTemplateAffineTransform.SetMatrix(subjectToTemplateSimilarityTransform.GetMatrix())
    subjectToTemplateAffineTransform.SetScale(subjectToTemplateSimilarityTransform.GetScale())
    subjectToTemplateAffineTransform.SetCenter(subjectToTemplateSimilarityTransform.GetCenter())
    subjectToTemplateAffineTransform.SetTranslation(subjectToTemplateSimilarityTransform.GetTranslation())

    subjectToTemplateAffineTransform = RegisterImages(headMask, templateHeadMask, subjectToTemplateAffineTransform, fixedMask=None, movingMask=None, useSDFs=False, verbose=True)

    resampledHeadImage = sitk.Resample(headMask, templateHeadMask, subjectToTemplateAffineTransform.GetInverse(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8)

    ## Extract the rotation from the affine transform
    newLandmarks = vtk.vtkPolyData()
    newLandmarks.DeepCopy(landmarks)

    # landmarksRegisteredByOldMethod = vtk.vtkPolyData()
    # landmarksRegisteredByOldMethod.DeepCopy(landmarks)

    for p in range(landmarks.GetNumberOfPoints()):
        coords = np.array(newLandmarks.GetPoint(p))
        tCoords = subjectToTemplateAffineTransform.TransformPoint(coords)
        newLandmarks.GetPoints().SetPoint(p, tCoords[0], tCoords[1], tCoords[2])

        # coords = np.array(landmarksRegisteredByOldMethod.GetPoint(p))
        # tCoords = toTemplateRigidTransform.TransformPoint(coords)
        # landmarksRegisteredByOldMethod.GetPoints().SetPoint(p, tCoords[0], tCoords[1], tCoords[2])

    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(os.path.join(outputPath, 'newLandmarks.vtp'))
    # writer.SetInputData(newLandmarks)
    # writer.Update()

    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName(os.path.join(outputPath, 'landmarksRegisteredByOldMethod.vtp'))
    # writer.SetInputData(landmarksRegisteredByOldMethod)
    # writer.Update()

    npLandmarks = np.zeros((landmarks.GetNumberOfPoints(), 3), dtype=np.float64)
    npNewLandmarks = np.zeros((newLandmarks.GetNumberOfPoints(), 3), dtype=np.float64)
    # nplandmarksRegisteredByOldMethod = np.zeros((landmarksRegisteredByOldMethod.GetNumberOfPoints(), 3), dtype=np.float64)
    for p in range(landmarks.GetNumberOfPoints()):
        npLandmarks[p,:] = landmarks.GetPoint(p)
        npNewLandmarks[p,:] = newLandmarks.GetPoint(p)
        # nplandmarksRegisteredByOldMethod[p,:] = landmarksRegisteredByOldMethod.GetPoint(p)

    #### attemtp with rigid registration:
    toTemplateRigidTransformFixed = AlignLandmarksWithTemplate(landmarks, newLandmarks, scaling=False)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetReferenceImage(segmentationTemplate)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(toTemplateRigidTransformFixed.GetInverse())
    ctImageRigidFixed = resampler.Execute(ctImage_masked)

    transform = sitk.AffineTransform(3)
    transform.SetIdentity()

    resampledCTImageFixed = sitk.Resample(ctImageRigidFixed, templateImage, transform.GetInverse(), sitk.sitkLinear)

    ##############################################
    ##############################################
    ##############################################

    # outputs
    # ImageArray = sitk.GetArrayFromImage(resampledHeadMask)
    ImageArray = sitk.GetArrayFromImage(resampledCTImageFixed)
    ImageArray[ImageArray > maxQuantile] = maxQuantile
    ImageArray = ImageArray/maxQuantile

    return ImageArray
