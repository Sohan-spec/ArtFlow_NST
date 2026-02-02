import utils.utils as utils
from utils.video_utils import create_video_from_results
import torch
from torch.optim import LBFGS,Adam
from torch.autograd import Variable
import numpy as np
import os
import argparse

def build_loss(neural_net,optimizing_img,target_representations,content_feature_maps_index,style_feature_maps_indices,config):
    target_content_representation=target_representations[0]
    target_style_representation=target_representations[1]
    
    current_set_of_feature_maps=neural_net(optimizing_img)
    current_content_representation=current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss=torch.nn.MSELoss(reduction='mean')(target_content_representation,current_content_representation)
    
    style_loss=0.0
    current_style_representation = []
    for cnt, x in enumerate(current_set_of_feature_maps):
        if cnt in style_feature_maps_indices:
            gram = utils.gram_matrix(x)
            current_style_representation.append(gram)
    for  (gram_target, gram_current) in zip(target_style_representation, current_style_representation):
        style_loss +=torch.nn.MSELoss(reduction='sum')(gram_target[0], gram_current[0]) #sum is less penalizign than mean, and this for me proved to produce better results
    style_loss/=len(target_style_representation)
    tv_loss=utils.total_variation_loss(optimizing_img)
    total_loss=config['content_weight']*content_loss + config['style_weight']*style_loss+ config['tv_weight']*tv_loss
    
    return total_loss,content_loss,style_loss,tv_loss

def make_tuning_step(neural_net,optimizer,target_representations,content_feature_maps_index,style_feature_maps_indices,config):
    def tuning_step(optimizing_img): #because lbfgs optimizer expects a function to be called repeatedly
        total_loss,content_loss,style_loss,tv_loss=build_loss(neural_net,optimizing_img,target_representations,content_feature_maps_index,style_feature_maps_indices,config)
        total_loss.backward() #this bascially back propagates and computes gradients, doesn't change anything
        
        optimizer.step() #this takes the gradients from the back prop and then changes the pixels accordingly, also in this we don't actually update weigths as the entire model is frozen, we just have to match current feature maps of the target image to feature maps of the content image, match the gram matrix of the target image to the gram matrix of the style image and voila you get your style done.
        
        optimizer.zero_grad() #resetsx the gradients
        return total_loss,content_loss,style_loss,tv_loss
    return tuning_step



def neural_style_transfer(config):
    content_image_path=os.path.join(config['content_images_dir'],config['content_img_name'])
    style_image_path=os.path.join(config['style_images_dir'],config['style_img_name'])
    
    output_dir_name= 'All_' +os.path.split(content_image_path)[1].split('.')[0]+'_'+os.path.split(style_image_path)[1].split('.')[0]
    dump_path=os.path.join(config['output_img_dir'],output_dir_name)
    os.makedirs(dump_path,exist_ok=True)
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    content_image=utils.prepare_image(content_image_path,config['height'],device)
    style_image=utils.prepare_image(style_image_path,config['height'],device)
    
    if config['initial_method']=='random':
        random_noise_image=np.random.normal(loc=0,scale=30.0,size=content_image.shape).astype(np.float32)
        initial_img=torch.from_numpy(random_noise_image).float().to(device)
    elif config['initial_method']=='content':
        initial_img=content_image
    else:
        style_image_resized=utils.prepare_image(style_image_path,np.asarray(content_image.shape[2:]),device)
        initial_img=style_image_resized
    
    optimizing_img=Variable(initial_img,requires_grad=True)
    neural_net,content_feature_maps_index_name,style_feature_maps_indices_names=utils.prepare_model(config['model'],device)
    print(f"using {config['model']}")
    
    content_img_set_of_feature_maps=neural_net(content_image)
    style_img_set_of_feature_maps=neural_net(style_image)
    
    target_content_representation=content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation=[utils.gram_matrix(x) for cnt,x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations=[target_content_representation,target_style_representation]
    
    num_of_iterations={"lbfgs":600} 
    if config['optimizer']=='lbfgs':
        optimizer=LBFGS((optimizing_img,),max_iter=num_of_iterations['lbfgs'],line_search_fn='strong_wolfe')
        cnt=0
        
        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss,content_loss,style_loss,tv_loss=build_loss(neural_net,optimizing_img,target_representations,content_feature_maps_index_name[0],style_feature_maps_indices_names[0],config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f"using LBFGS, iteration : {cnt:03}, total_loss : {total_loss.item():12.4f}, content_loss : {config['content_weight'] * content_loss.item():12.4f}, total_variaton loss : {config['tv_weight'] * tv_loss.item():12.4f}")
                utils.save_and_maybe_display(optimizing_img,dump_path,config,cnt,num_of_iterations[config['optimizer']])
                if config['should_display']==True:
                    if (cnt%50)==0:
                        utils.display_feature_images(optimizing_img)
            cnt+=1
            return total_loss
        optimizer.step(closure)
    if config['saving_freq'] > 0:
        print("yo vid comin up buddy, enjoy the slow mo")
        create_video_from_results(dump_path, config['img_format'])
    else:
        print("skipped vid gen as freq=-1")
    return dump_path


if __name__ == "__main__":
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--content_img_name",type=str,help="choose any image from the data/content-images or feel free to add your own images there",default='taj_mahal.jpg')
    parser.add_argument("--style_img_name",type=str,help="choose any image from the data/style-images or feel free to add your own images there",default='vg_starry_night.jpg')
    parser.add_argument("--height",type=int,help="The height you want to resize your image to (ex: like 224,256) etc",default=400)
    parser.add_argument("--content_weight",type=float,help="weights for content loss",default=1e5)
    parser.add_argument("--style_weight",type=float,help="weights for style loss",default=3e2)
    parser.add_argument("--tv_weight",type=float,help="weights for total variation loss",default=1e0)
    parser.add_argument("--optimizer",type=str,help="i only choose LBFGS as its the best, no choice sorry :)",default='lbfgs')
    parser.add_argument("--model",type=str,help="choose your model",choices=['vgg16','vgg19'],default='vgg19')
    parser.add_argument("--initial_method",type=str,help="choose your initial method, random noise image, the content image or the style image",choices=['random','content','style'],default='content')
    parser.add_argument("--saving_freq",type=int,help="how often your images get saved (cycles), choose -1 for only saving the last output image",default=10)
    parser.add_argument("--should_display",type=bool,help="do you want the output of features to be displayed",default=False)
    args=parser.parse_args()
    
    optimization_config=dict()
    for arg in vars(args):
        optimization_config[arg]=getattr(args,arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format
    results_path=neural_style_transfer(optimization_config)
        
    
    