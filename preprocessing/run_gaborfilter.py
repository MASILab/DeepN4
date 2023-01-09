import numpy as np
from scipy import ndimage
import nibabel as nib

def loggabor3d(shape, sigma, frequency, theta, phi):
    """
    Create a 3D log-Gabor filter with the specified shape, standard deviation,
    frequency, orientation, and phase.
    """
    sigma_x = sigma[0]
    sigma_y = sigma[1]
    sigma_z = sigma[2]
    
    # Create 3D meshgrid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    
    # Convert x, y, z to spherical coordinates
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / rho)
    
    # Rotate coordinates by the specified angle
    theta = theta - theta
    phi = phi - phi
    
    # Convert back to Cartesian coordinates
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    
    # Compute the log-Gabor filter
    loggabor = np.exp(-0.5 * ((x / sigma_x)**2 + (y / sigma_y)**2 + (z / sigma_z)**2)) * np.cos(2 * np.pi * frequency * x + phi)
    return loggabor

with open('cbtrain_100ds.csv', 'r') as f:
     for line in f:
        inputfile = line.split(',')[1]
        img = nib.load(inputfile)
        data = img.get_fdata()

        # filter
        theta = 0
        phi = 0
        loggabor = loggabor3d((24,24,24), (4, 4, 4), 0, theta, phi)
        filtered_image1 = ndimage.convolve(data, loggabor)
        loggabor = loggabor3d((16,16,16), (4, 4, 4), 0, theta, phi)
        filtered_image2 = ndimage.convolve(data, loggabor)
        loggabor = loggabor3d((8,8,8), (4, 4, 4), 0, theta, phi)
        filtered_image3 = ndimage.convolve(data, loggabor)

        output_file1 = '/home/local/VANDERBILT/kanakap/deepN4_data/filtered1/' + inputfile.split('/')[-1]
        output_file2 = '/home/local/VANDERBILT/kanakap/deepN4_data/filtered2/' + inputfile.split('/')[-1]
        output_file3 = '/home/local/VANDERBILT/kanakap/deepN4_data/filtered3/' + inputfile.split('/')[-1]

        # save the image 
        nii1 = nib.Nifti1Image(filtered_image1, affine=img.affine, header=img.header)
        nib.save(nii1, output_file1)

        nii2 = nib.Nifti1Image(filtered_image2, affine=img.affine, header=img.header)
        nib.save(nii2, output_file2)

        nii3 = nib.Nifti1Image(filtered_image3, affine=img.affine, header=img.header)
        nib.save(nii3, output_file3)

        # write to training csv list 
        with open('rnet_train.csv','w') as p:
            bias_file = '/home/local/VANDERBILT/kanakap/deepN4_data/bias/' + inputfile.split('/')[-1]
            correct_file = '/home/local/VANDERBILT/kanakap/deepN4_data/labels/' + inputfile.split('/')[-1]
            p.write("%s,%s,%s,%s,%s,%s\n" % (correct_file,inputfile,bias_file,output_file1,output_file2,output_file3))

