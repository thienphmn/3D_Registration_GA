import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os, time


# === Loading and Saving images ===
def load_image(path: str, modality: str, wl: int=40, ww: int=400, smooth: bool=True) -> sitk.Image:
    """
        Load images and apply appropriate preprocessing for image registration.
        Args:
            path: path to image file
            modality: either "CT" or "MR"
            wl: window level
            ww: window width
            smooth: whether to smooth the image
        Returns:
            sitk.Image
        """
    # Load image
    image = sitk.ReadImage(path)

    if modality == "CT":
        # Window and Normalize CT
        lower = wl - ww // 2
        upper = wl + ww // 2
        image = sitk.IntensityWindowing(image, windowMinimum=lower, windowMaximum=upper,
                                        outputMinimum=0.0, outputMaximum=255.0)
    elif modality == "MR":
        # Normalize MR
        image = sitk.RescaleIntensity(image, outputMinimum=0.0, outputMaximum=255.0)

        # Pad Image
        image = sitk.ConstantPad(image,
                                 padLowerBound=[10,10,10],
                                 padUpperBound=[10,10,10],
                                 constant=0.0)

    if smooth:
        # Reduce noise with gaussian filter
        image = sitk.SmoothingRecursiveGaussian(image, sigma=[1,1,1], normalizeAcrossScale=True)
        image = sitk.RescaleIntensity(image, outputMinimum=0.0, outputMaximum=255.0)

    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def show_sitk_img_info(img: sitk.Image) -> None:
    """
    Get important information from sitk.Image.
    Args:
        img: a sitk.Image object
    Returns:
        None
    """
    pixel_type = img.GetPixelIDTypeAsString()
    origin = img.GetOrigin()
    dimensions = img.GetSize()
    spacing = img.GetSpacing()
    direction = img.GetDirection()

    info = {'Pixel Type': pixel_type, 'Dimensions': dimensions, 'Spacing': spacing,
            'Origin': origin, 'Direction': direction}
    for k, v in info.items():
        print(f' {k} : {v}')


def convert_mhd_to_dicom(path_to_mhd: str, modality: str) -> None:
    """Convert .mhd file to .dicom series for teaching purposes."""
    # 1. Read image
    image = sitk.ReadImage(path_to_mhd)

    # 2. Prepare output folder for DICOM slices
    if modality == "CT":
        output_dir = "/Users/thien/Desktop/ct"
    elif modality == "MR":
        output_dir = "/Users/thien/Desktop/mr"
    else:
        raise NotImplementedError
    os.makedirs(output_dir, exist_ok=True)

    # 3. Generate UIDs and shared identifiers
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Shared UIDs across all slices in the series
    series_uid = "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time
    study_uid = "1.2.826.0.1.3680043.2.1125." + modification_date + ".2" + modification_time
    frame_of_ref_uid = "1.2.826.0.1.3680043.2.1125." + modification_date + ".3" + modification_time

    # Add Accession Number - Use a simple, shared identifier for the study
    accession_number = "RIRE" + modification_date + modification_time

    # 4. Create a DICOM writer
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # 5. Get spacing and origin for proper spatial encoding
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    # 6. Calculate window/level for proper display
    image_array = sitk.GetArrayFromImage(image)
    min_val = float(np.min(image_array))
    max_val = float(np.max(image_array))

    # Window Width and Window Center for display
    window_width = max_val - min_val
    window_center = min_val + window_width / 2.0

    # 7. Loop through slices in the 3D volume
    for i in range(image.GetDepth()):
        slice_i = image[:, :, i]

        # Copy spatial information
        slice_i.SetSpacing([spacing[0], spacing[1]])
        slice_i.SetDirection([direction[0], direction[1], direction[3], direction[4]])

        # Calculate position for this slice
        slice_origin = [
            origin[0] + direction[2] * i * spacing[2],
            origin[1] + direction[5] * i * spacing[2],
            origin[2] + direction[8] * i * spacing[2]
        ]
        slice_i.SetOrigin([slice_origin[0], slice_origin[1]])

        # Generate unique SOP Instance UID for this slice
        sop_uid = "1.2.826.0.1.3680043.2.1125." + modification_date + ".4" + modification_time + "." + str(i)

        # --- CRITICAL: Tags that MUST match across all slices ---
        slice_i.SetMetaData("0020|000d", study_uid)  # Study Instance UID
        slice_i.SetMetaData("0020|000e", series_uid)  # Series Instance UID
        slice_i.SetMetaData("0020|0052", frame_of_ref_uid)  # Frame of Reference UID
        slice_i.SetMetaData("0020|0011", "1")  # Series Number
        slice_i.SetMetaData("0008|103e", "RIRE Study")  # Series Description
        slice_i.SetMetaData("0010|0010", "RIRE^Patient")  # Patient Name
        slice_i.SetMetaData("0010|0020", "12345")  # Patient ID
        slice_i.SetMetaData("0008|0060", modality)  # Modality
        slice_i.SetMetaData("0008|0050", accession_number)  # <--- ADD THIS: Accession Number

        # --- Tags specific to this slice ---
        if modality == "CT":
            sop_class_uid = "1.2.840.10008.5.1.4.1.1.2"
        elif modality == "MR":
            sop_class_uid = "1.2.840.10008.5.1.4.1.1.4"
        else:
            raise NotImplementedError
        slice_i.SetMetaData("0008|0016", sop_class_uid)  # SOP Class UID
        slice_i.SetMetaData("0008|0018", sop_uid)  # SOP Instance UID
        slice_i.SetMetaData("0020|0013", str(i + 1))  # Instance Number

        # --- Spatial positioning tags ---
        slice_i.SetMetaData("0018|0050", str(spacing[2]))  # Slice Thickness
        slice_i.SetMetaData("0018|0088", str(spacing[2]))  # Spacing Between Slices
        slice_i.SetMetaData("0028|0030", f"{spacing[0]}\\{spacing[1]}")  # Pixel Spacing
        slice_i.SetMetaData("0020|0032",
                            f"{slice_origin[0]}\\{slice_origin[1]}\\{slice_origin[2]}")  # Image Position Patient
        slice_i.SetMetaData("0020|0037", "1\\0\\0\\0\\1\\0")  # Image Orientation Patient
        slice_i.SetMetaData("0020|1041", str(slice_origin[2]))  # Slice Location

        # --- Display windowing tags (CRITICAL for proper brightness) ---
        slice_i.SetMetaData("0028|1050", str(window_center))  # Window Center
        slice_i.SetMetaData("0028|1051", str(window_width))  # Window Width
        slice_i.SetMetaData("0028|0106", str(min_val))  # Smallest Image Pixel Value
        slice_i.SetMetaData("0028|0107", str(max_val))  # Largest Image Pixel Value

        # --- Date/Time tags ---
        slice_i.SetMetaData("0008|0012", modification_date)  # Instance Creation Date
        slice_i.SetMetaData("0008|0013", modification_time)  # Instance Creation Time
        slice_i.SetMetaData("0008|0020", modification_date)  # Study Date
        slice_i.SetMetaData("0008|0030", modification_time)  # Study Time
        slice_i.SetMetaData("0008|0021", modification_date)  # Series Date
        slice_i.SetMetaData("0008|0031", modification_time)  # Series Time

        # 8. Write slice to DICOM file
        writer.SetFileName(os.path.join(output_dir, f"slice_{i:03d}.dcm"))
        writer.Execute(slice_i)

    print("DICOM series written to:", output_dir)


def smooth_and_resample(image: sitk.Image, shrink: int|list):
    """
    Args:
        image: SimpleITK image.
        shrink: shrink factor for anisotropic shrinking in x and y directions
    Return:
        Smoothed and downsampled image with preserved metadata.
    """
    # Normalize shrink factor to 3-element list
    if isinstance(shrink, int):
        shrink = [shrink, shrink, 1]

    # Gaussian sigma per axis: classic antialiasing (≈ shrink/2)
    spacing = image.GetSpacing()
    sigma = []
    for index, factor in enumerate(shrink):
        if factor > 1:
            sigma.append((factor / 2.0) * spacing[index])
        else:
            sigma.append(1e-6)

    # Blur first (metadata preserved automatically)
    blurred = sitk.SmoothingRecursiveGaussian(image1=image, sigma=sigma, normalizeAcrossScale=True)

    # Downsample using inbuilt SimpleITK Shrink (handles spacing + origin + direction)
    return sitk.Shrink(blurred, shrink)


# === Visualizing Images ===
def show_slices(image: sitk.Image) -> None:
    """
    Show all axial slices of a given image.
    Args:
        image: sitk.Image object
    Returns:
        None
    """
    array = sitk.GetArrayFromImage(image)
    for SLICE in range(len(array)):
        plt.figure(figsize=(7, 7))
        plt.imshow(array[SLICE], cmap="grey", vmin=array.min(), vmax=array.max())
        plt.show()
        plt.close()


def show_registered_images(fixed: sitk.Image, moving: sitk.Image, grey_fusion:bool=False) -> None:
    """
    Overlay two 3D images, where the fixed one is green and the moving one is magenta.
    Args:
        fixed: sitk.Image object
        moving: sitk.Image object
        transform: Transformation that is applied to the moving image
        grey_fusion: whether to apply grey fusion
    Returns:
        None
    """
    resampled_moving = sitk.Resample(image1=moving,
                                     referenceImage=fixed,
                                     transform=sitk.Transform(),
                                     interpolator = sitk.sitkLinear,
                                     defaultPixelValue = 0)

    # Get arrays from images
    fixed_array = sitk.GetArrayFromImage(fixed).astype(np.uint8) # shape: [x,y,z]
    moving_array = sitk.GetArrayFromImage(resampled_moving).astype(np.uint8)

    # Visualize slice by slice
    for sliding_window in range(fixed_array.shape[0]):
        fused = np.stack([moving_array[sliding_window], fixed_array[sliding_window], moving_array[sliding_window]], axis=2)
        plt.figure(figsize=(7, 7))
        plt.imshow(fused)
        plt.show()
        plt.close()

    if grey_fusion:
        # Alpha Blending in Grey Fusion
        center_slice = len(fixed_array) // 2
        for i in range(11):
            alpha = 0.1 * i
            plt.imshow(fixed_array[center_slice], cmap='gray', alpha=1)
            plt.imshow(moving_array[center_slice], cmap='gray', alpha=alpha)
            plt.axis('off')
            plt.show()
            plt.close()


# === 3D Image Transformations ===
def create_3d_rigid_transform(image: sitk.Image, tx:float, ty:float, tz:float, rx:float, ry:float, rz:float) -> sitk.Transform:
    """
    Create a 3D rigid transformation with translation and rotation.
    Rotation is parameterized using 3 values (rx, ry, rz) that are converted
    to a versor (unit quaternion). This parameterization is suitable for
    continuous optimization in Genetic Algorithms.
    Args:
        image: sitk.Image object
        tx: translation in x-axis
        ty: translation in y-axis
        tz: translation in z-axis
        rx: x-component of rotation vector
        ry: y-component of rotation vector
        rz: z-component of rotation vector
    Returns:
        sitk.Transform: Transformation to be applied to an image.
    """
    # Convert degrees to radians
    rx, ry, rz = np.deg2rad([rx, ry, rz])

    # Get the center of the image (center of rotation)
    center = get_image_center(image)

    # Create the versor from rotation parameters
    versor = rotation_params_to_versor(rx, ry, rz)

    # Create VersorRigid3DTransform
    transform = sitk.VersorRigid3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(versor)
    transform.SetTranslation([tx, ty, tz])
    return transform


def get_image_center(image: sitk.Image) -> tuple[float,float,float]:
    """
    Calculate the physical center of an image as physical coordinates.
    Args:
        image: sitk.Image object
    Returns:
        Three floats describing the physical center of an image.
    """
    size = image.GetSize()

    # Center in index coordinates
    center_index = [(dimension_size - 1) / 2.0 for dimension_size in size]

    # Convert to physical coordinates (accounts for spacing, origin, direction)
    center_physical = image.TransformContinuousIndexToPhysicalPoint(center_index)
    return center_physical


def rotation_params_to_versor(rx:float, ry:float, rz:float) -> tuple[float,...]:
    """
    Convert 3 rotation parameters to a versor (unit quaternion).
    Method: Treat (rx, ry, rz) as an axis-angle representation where
    the axis is (rx, ry, rz) normalized and angle is the magnitude.
    Args:
        rx: x-component of rotation vector
        ry: y-component of rotation vector
        rz: z-component of rotation vector
    Return:
        Versor describing the rotation to feed into sitk.Transform
    """
    # Calculate the magnitude (rotation angle in radians)
    angle = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Handle the zero rotation case
    if angle < 1e-10:
        return 0.0, 0.0, 0.0, 1.0  # Identity rotation

    # Normalize to get the rotation axis
    axis_x = rx / angle
    axis_y = ry / angle
    axis_z = rz / angle

    # Convert axis-angle to quaternion
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    qx = axis_x * sin_half
    qy = axis_y * sin_half
    qz = axis_z * sin_half
    qw = cos_half

    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    return qx / norm, qy / norm, qz / norm, qw / norm


def transform_image(moving:sitk.Image, fixed: sitk.Image,
                    tx:float=0, ty:float=0, tz:float=0, rx:float=0, ry:float=0, rz:float=0) -> sitk.Image:
    """
    Transform an image using a 3D rigid transformation.
    Args:
        fixed: sitk. Image object
        moving: sitk. Image object
        tx: translation in x-axis
        ty: translation in y-axis
        tz: translation in z-axis
        rx: x-component of rotation vector
        ry: y-component of rotation vector
        rz: z-component of rotation vector
    Returns:
        Transformed image as sitk.Image object
    """
    # Create the transformation
    transform = create_3d_rigid_transform(image=fixed, tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz)

    # Apply the transformation using resampling
    resampled_image = sitk.Resample(image1=moving,
                                    referenceImage=fixed,
                                    transform=transform,
                                    interpolator=sitk.sitkLinear,
                                    defaultPixelValue=0.0)
    return resampled_image


# === Similarity Score Calculation ===
def calculate_mutual_information(fixed: sitk.Image, resampled: sitk.Image, num_bins: int = 50) -> float:
    """
    Args:
        fixed: reference image
        resampled: moving image that has been resampled to match the fixed image's grid
        num_bins: number of bins in the histogram
    Returns:
        Mutual information between fixed and resampled image
    """
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=num_bins)
    registration.SetInitialTransform(sitk.Transform())
    registration.SetInterpolator(sitk.sitkLinear)


    mutual_information = -registration.MetricEvaluate(fixed, resampled)
    return mutual_information



if __name__ == "__main__":
    ct = load_image("/Users/thien/Documents/Development/3D_Image_Reg/dataset/RIRE/ct/training_001_ct.mhd","CT")
    pd = load_image("/Users/thien/Documents/Development/3D_Image_Reg/dataset/RIRE/mr_PD_rectified/training_001_mr_PD_rectified.mhd","MR")
    t1 = load_image("/Users/thien/Documents/Development/3D_Image_Reg/dataset/RIRE/mr_T1_rectified/training_001_mr_T1_rectified.mhd", "MR")
    t2 = load_image("/Users/thien/Documents/Development/3D_Image_Reg/dataset/RIRE/mr_T2_rectified/training_001_mr_T2_rectified.mhd", "MR")

    show_sitk_img_info(ct)
    print()
    show_sitk_img_info(pd)
    print()
    show_sitk_img_info(t1)
    print()
    show_sitk_img_info(t2)
    print()

    show_registered_images(ct, pd)
    # show_registered_images(ct, t1)
    # show_registered_images(ct, t2)
