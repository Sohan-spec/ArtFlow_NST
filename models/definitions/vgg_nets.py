from collections import namedtuple
import torch
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights
    
class Vgg16(torch.nn.Module): #also this isn't the original papers vgg16 conv style this is the modern relu style (inspired from vgg19) which is better than the original paper's vgg16
    def __init__(self,requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.layer_names=['relu1_1','relu2_1','relu2_2','relu3_1','relu3_2','relu4_1','relu4_2','relu4_3','relu5_1']# added relu4_2 in my version
        self.content_feature_maps_index=4 #this is like in the middle (not to deep not to early) sooo this would capture semantics properly
        self.style_feature_maps_indices=list(range(len(self.layer_names)))
        self.vgg_outputs=namedtuple("VggOutputs", self.layer_names)
        
        self.conv1_1=vgg_pretrained_features[0]
        self.relu1_1=vgg_pretrained_features[1]
        self.conv1_2=vgg_pretrained_features[2]
        self.relu1_2=vgg_pretrained_features[3]
        self.max_pooling1=vgg_pretrained_features[4]
        self.conv2_1=vgg_pretrained_features[5]
        self.relu2_1=vgg_pretrained_features[6]
        self.conv2_2=vgg_pretrained_features[7]
        self.relu2_2=vgg_pretrained_features[8]
        self.max_pooling2=vgg_pretrained_features[9]
        self.conv3_1=vgg_pretrained_features[10]
        self.relu3_1=vgg_pretrained_features[11]
        self.conv3_2=vgg_pretrained_features[12]
        self.relu3_2=vgg_pretrained_features[13]
        self.conv3_3=vgg_pretrained_features[14]
        self.relu3_3=vgg_pretrained_features[15]
        self.max_pooling3=vgg_pretrained_features[16]
        self.conv4_1=vgg_pretrained_features[17]
        self.relu4_1=vgg_pretrained_features[18]
        self.conv4_2=vgg_pretrained_features[19]
        self.relu4_2=vgg_pretrained_features[20]
        self.conv4_3=vgg_pretrained_features[21]
        self.relu4_3=vgg_pretrained_features[22]
        self.max_pooling4=vgg_pretrained_features[23]
        self.conv5_1 = vgg_pretrained_features[24]
        self.relu5_1 = vgg_pretrained_features[25]
        self.conv5_2 = vgg_pretrained_features[26]
        self.relu5_2 = vgg_pretrained_features[27]
        self.conv5_3 = vgg_pretrained_features[28]
        self.relu5_3 = vgg_pretrained_features[29]
        self.max_pooling5 = vgg_pretrained_features[30]
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
                
    def forward(self,x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        relu1_1 = x
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        relu1_2 = x
        x = self.max_pooling1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        relu2_1 = x
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        relu2_2 = x
        x = self.max_pooling2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        relu3_1 = x
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        relu3_2 = x
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        relu3_3 = x
        x = self.max_pooling3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        relu4_1 = x
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        relu4_2 = x
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        relu4_3 = x
        x = self.max_pooling4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        relu5_1 = x
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        relu5_2 = x
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        relu5_3 = x
        x = self.max_pooling5(x)
        out=self.vgg_outputs(relu1_1, relu2_1, relu2_2, relu3_1, relu3_2, relu4_1,relu4_2, relu4_3, relu5_1)
        return out
    #removed all the conv layers here as the model doesn't expose those conv layers rather just exposes the relu feature mapss
    
        
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.layer_names=['conv1_1','conv2_1','conv3_1','conv4_1','conv4_2','conv5_1'] # this is acc to the original paper of vgg19
        self.content_feature_maps_index = 4
        self.style_feature_maps_indices = [0,1,2,3,5] #4 i.e conv4_2 was removed in the original paper too
        self.vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        self.conv1_1 = vgg[0]
        self.relu1_1 = vgg[1]
        self.conv1_2 = vgg[2]
        self.relu1_2 = vgg[3]
        self.pool1 = vgg[4]

        self.conv2_1 = vgg[5]
        self.relu2_1 = vgg[6]
        self.conv2_2 = vgg[7]
        self.relu2_2 = vgg[8]
        self.pool2 = vgg[9]

        self.conv3_1 = vgg[10]
        self.relu3_1 = vgg[11]
        self.conv3_2 = vgg[12]
        self.relu3_2 = vgg[13]
        self.conv3_3 = vgg[14]
        self.relu3_3 = vgg[15]
        self.conv3_4 = vgg[16]
        self.relu3_4 = vgg[17]
        self.pool3 = vgg[18]

        self.conv4_1 = vgg[19]
        self.relu4_1 = vgg[20]
        self.conv4_2 = vgg[21]
        self.relu4_2 = vgg[22]
        self.conv4_3 = vgg[23]
        self.relu4_3 = vgg[24]
        self.conv4_4 = vgg[25]
        self.relu4_4 = vgg[26]
        self.pool4 = vgg[27]

        self.conv5_1 = vgg[28]
        self.relu5_1 = vgg[29]
        self.conv5_2 = vgg[30]
        self.relu5_2 = vgg[31]
        self.conv5_3 = vgg[32]
        self.relu5_3 = vgg[33]
        self.conv5_4 = vgg[34]
        self.relu5_4 = vgg[35]
        self.pool5 = vgg[36]

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        x = self.conv1_1(x)
        conv1_1 = x

        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        conv2_1 = x
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        conv3_1 = x
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.conv3_4(x)
        x = self.relu3_4(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        conv4_1 = x 
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        conv4_2 = x
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x = self.conv4_4(x)
        x = self.relu4_4(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        conv5_1 = x
        out=self.vgg_outputs(conv1_1,conv2_1,conv3_1,conv4_1,conv4_2,conv5_1)

        return out

        