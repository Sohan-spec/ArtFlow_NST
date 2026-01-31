import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt

from models.definitions.vgg_nets import Vgg19, Vgg16

style_layer_weights = [1.0, 1.0, 1.0, 0.5, 0.25, 0.1, 0.1, 0.1, 0.05]
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path,target_shape=None):
    if not os.path.exists(img_path):
        raise Exception("Image path not found")
    img=cv.imread(img_path)[:,:,::-1]
    
    if target_shape is not None:
        if isinstance(target_shape,int) and target_shape !=-1:
            current_height,current_width=img.shape[:2]
            new_height=target_shape
            new_width=int(current_width*(new_height/current_height))
            img=cv.resize(img,(new_width,new_height),interpolation=cv.INTER_CUBIC)
        else:
            img=cv.resize(img,(target_shape[1],target_shape[0]),interpolation=cv.INTER_CUBIC)
    
    img=img.astype(np.float32)
    img/=255.0
    return img

def prepare_image(img_path,target_shape,device):
    img=load_image(img_path,target_shape)
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255,std=IMAGENET_STD_NEUTRAL)
    ])
    img=transform(img).to(device).unsqueeze(0)
    return img

def save_image(img,img_path):
    if len(img.shape)==2:
        img=np.stack((img,)*3,axis=-1)
    cv.imwrite(img_path,img[:,:,::-1])

def generate_out_image_name(config):
    prefix=os.path.basename(config['content_img_name']).split('.')[0]+'_'+os.path.basename(config['style_img_name']).split('.')[0]
    
    if 'reconstruct_script' in config:
        suffix=f"_o_{config['optimizer']}_h_{config['height']}_m_{config['model']}{config['img_format'][1]}"
    else:
        suffix=f"_o_{config['optimizer']}_i_{config['init_method']}_h_{str(config['height'])}_m_{config['model']}_cw_{config['content_weight']}_sw_{config['style_weight']}_tv_{config['tv_weight']}{config['img_format'][1]}"
    return prefix + suffix
        
        
        
def save_and_maybe_display(optimizing_img,dump_path,config,img_id,num_of_iterations,should_display=False):
    saving_frequency=config['saving_freq']
    out_img=optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img=np.moveaxis(out_img,0,2)
    
    if img_id == num_of_iterations-1 or (saving_frequency > 0 and img_id%saving_frequency==0):
        img_format=config['img_format']
        out_img_name=str(img_id).zfill(img_format[0])+img_format[1] if saving_frequency !=-1 else generate_out_image_name(config)
        dump_img=np.copy(out_img)
        dump_img+=np.array(IMAGENET_MEAN_255).reshape((1,1,3))
        dump_img=np.clip(dump_img,0,255).astype('uint8')
        cv.imwrite(os.path.join(dump_path,out_img_name),dump_img[:,:,::-1])
    
    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


def get_uint8_range(x):
    if isinstance(x,np.ndarray):
        x-=np.min(x)
        x/=(np.max(x)+1e-8)
        x*=255
        return x
    else:
        raise ValueError(f'Expected numpy array but got{type(x)}')
            

def prepare_model(model,device):
    model=model.lower()
    if model =='vgg16':
        model=Vgg16(requires_grad=False,show_progress=True)
    elif model=='vgg19':
        model=Vgg19(requires_grad=False,show_progress=True)
    else:
        raise ValueError(f"{model} was not found")
    
    content_feature_maps_index=model.content_feature_maps_index
    style_feature_maps_indices=model.style_feature_maps_indices
    layer_names=model.layer_names
    
    content_layer_info=(content_feature_maps_index,layer_names[content_feature_maps_index])
    style_layers_info=(style_feature_maps_indices,layer_names)
    model = model.to(device)
    for p in model.parameters():
        assert p.device == torch.device(device)
    return model.to(device).eval(), content_layer_info,style_layers_info

def gram_matrix(x,should_normalize=True):
    (b,ch,h,w)=x.size()
    features=x.reshape(b,ch,w*h)
    features_t=features.transpose(1,2)
    gram=features.bmm(features_t)
    if should_normalize:
        gram=gram/(ch*h*w)
    return gram

def total_variation(y):
    return torch.sum(torch.abs(y[:,:,:,:-1]-y[:,:,:,1:])) + torch.sum(torch.abs(y[:,:,:-1,:]-y[:,:,1:,:]))
'''this is like:
  for the first torch.abs [b,c,h,w] we're doing take all the Batches, all channels, and height(rows) but remove the first and last column (width), as 1st and last column don't have any left and right neighbours
  
  for the sencond one we're doing take all the batches, all channels, remove the first and last row (height) and take all the columns, same reason as the first one
'''
    
    