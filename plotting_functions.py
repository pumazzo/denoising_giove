import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from kspace_handling_functions import do_image

# Helper function to show a batch
def show_generator_output(sample_batched):
    """Show image with landmarks for a batch of samples."""
    k_batch, k_label_batch = sample_batched[0], sample_batched[1]
    
    # convert them to numpy and then to complex so they can be used with do_image
    k_batch = k_batch.detach().numpy()
    k_label_batch = k_label_batch.detach().numpy()
    # to complex 
    k_batch = k_batch[:,0,:,:] + 1j*k_batch[:,1,:,:]
    k_label_batch = k_label_batch[:,0,:,:] + 1j*k_label_batch[:,1,:,:]
    
    
    images_batch = do_image(k_batch)  
    label_batch = do_image(k_label_batch) 
    
    
    batch_size = len(images_batch)
    print(batch_size)
    if batch_size > 8:
        batch_size = 8
    im_size = images_batch.shape[2:3]
    

    f,axes = plt.subplots(batch_size,2,figsize=(15,30))
    for i in range(batch_size):
        axes[i,0].imshow(images_batch[i])
        axes[i,0].set_title("Low SNR")
        axes[i,1].imshow(label_batch[i])
        axes[i,1].set_title("high SNR")
        axes[i,0].axis('off')
        #axes[i,0].ioff()
        axes[i,1].axis('off')
        #axes[i,1].ioff()
        
        
def show_pred_batch(sample_batched,pred):
    """Show image with landmarks for a batch of samples."""
    k_batch, k_label_batch = sample_batched[0], sample_batched[1]
    
    # convert them to numpy and then to complex so they can be used with do_image
    k_batch = k_batch.detach().numpy()
    k_label_batch = k_label_batch.detach().numpy()
    # to complex 
    k_batch = k_batch[:,0,:,:] + 1j*k_batch[:,1,:,:]
    k_label_batch = k_label_batch[:,0,:,:] + 1j*k_label_batch[:,1,:,:]
    
    images_batch = do_image(k_batch)  
    label_batch = do_image(k_label_batch) 
    
    
    
    
    pred_image = pred[0].detach().numpy()
    pred_image = pred_image[:,0,:,:] + 1j*pred_image[:,1,:,:]
    
    pred_image = do_image(pred_image)
    
    out_image = pred[1].detach().numpy()
    
    
    batch_size = len(images_batch)
    print(batch_size)
    if batch_size > 8:
        batch_size = 8
    im_size = images_batch.shape[2:3]
    
    images_batch = np.squeeze(images_batch)
    label_batch = np.squeeze(label_batch)
    pred_image = np.squeeze(pred_image)
    out_image = np.squeeze(out_image)
    
    artifact = np.median(out_image,axis=0)
    #out_image=out_image-artifact
    
    f,axes = plt.subplots(batch_size,3,figsize=(20,30))
    axes[0,0].set_title("low SNR",fontsize=25)
    axes[0,1].set_title("High SNR",fontsize=25)
    #axes[0,2].set_title("K space predicted and transformed")
    axes[0,2].set_title("Network predictcion",fontsize=25)
    
    for i in range(batch_size):
        axes[i,0].imshow(images_batch[i])
        axes[i,1].imshow(label_batch[i])
        
        
        pred_image[i,39:41,63:65] = pred_image[i].min()
        out_image[i,39:41,63:65] = out_image[i].min()
        
        #axes[i,2].imshow(pred_image[i])
        axes[i,2].imshow(out_image[i],vmin=np.percentile(out_image.ravel(),90)) #np.max([0.75*out_image[i].max(),0.5]))
        
        
        axes[i,0].axis('off')
        #axes[i,0].ioff()
        axes[i,1].axis('off')
        axes[i,2].axis('off')
        #axes[i,3].axis('off')
        #axes[i,1].ioff()
    plt.tight_layout()
    plt.savefig("results_example.jpg")
    print(np.percentile(out_image.ravel(),60))
    print(0.75*out_image.max())
    plt.figure()
    plt.hist(label_batch.ravel(),100)
    plt.title("target image")
    plt.figure()
    plt.hist(out_image.ravel(),100)
    plt.title("out image")