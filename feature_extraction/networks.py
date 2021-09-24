import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable

'''
pre-trained ResNet
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet(nn.Module):
    '''
    Args: 
        feature_type: string, resnet101 or resnet152
    '''
    
    def __init__(self, feature_type='resnet152'):
        super().__init__()

        self.feature_type = feature_type
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if feature_type == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        elif feature_type == 'resnet152':
            resnet = models.resnet152(pretrained=True)
        else:
            raise Exception('No such ResNet!')
        
        resnet.float()
        resnet.to(device=device)
        resnet.eval()

        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
        '''
        rescale and normalize image, then pass it through ResNet
        '''

        x = self.transform(x)
        x = x.unsqueeze(0)  # add batch size dim
        x = Variable(x).to(device=device)
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)

        return res_pool5