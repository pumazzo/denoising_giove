# %%
import numpy as np
import matplotlib.pyplot as plt



def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(np.abs(sig)**2, axis))

from skimage.measure import block_reduce

def do_image(kspace):
    block_size = np.ones(len(kspace.shape),dtype="int")
    block_size[-1]=2
    block_size=tuple(block_size)
   # print(block_size)
    kspace=block_reduce(kspace, block_size=block_size, func=np.mean)
    image = ifftnd(kspace, [-1,-2])
    image = np.abs(image)
    
    

    return image
    
def plot_pair(X_test,y_test,vol=1,fetta=12):
    im_GT = do_image(y_test[vol,:,:,:,:])
    im = do_image(X_test[vol,:,:,:,:])
    im_GT = np.squeeze(im_GT)
    im = np.squeeze(im)
    f,axes=plt.subplots(1,2,figsize=(12,12))
    axes[0].imshow(im[fetta,:])
    axes[0].set_title("data")
    axes[1].imshow(im_GT[fetta,:])
    axes[1].set_title("target")


# %% normalization in k space


def normalize_image(vol):
    vol=(vol-vol.min())/(vol.max()-vol.min())
    
    return vol

def normalize_kspace_max(vol):
    vol = vol - vol.mean()# forse?
    norm = np.abs(vol).max()
    return vol/norm

def normalize_kspace_z(vol):
    x = np.concatenate([vol.real,vol.imag],axis=1)
    mean = np.mean(x,axis=(-2,-1),keepdims=True)
    std = np.std(x,axis=(-2,-1),keepdims=True)
    tmp = (x-mean)/std
    
    vol = tmp[:,0,:,:]+1j*tmp[:,1,:,:]
    vol = np.expand_dims(vol,axis=1)

    return vol

def norm_dataset(vol):
    #normed=np.apply_along_axis(normalize_kspace_z, 0, vol)
    normed = np.zeros(vol.shape,dtype=vol.dtype)
    for i in range(vol.shape[0]):
        
        #normed[i,:,:,:] = normalize_kspace_z(vol[i,:,:,:])
        normed[i,:,:,:] = normalize_kspace_max(vol[i,:,:,:])
    
    return normed






























