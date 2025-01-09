import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinear,custom_cnn, attention_layer
import timm
from backbone.resnet import resnet18,resnet34,resnet50,resnet101

def get_backbone(args, backbone_type, pretrained=False):
    name = backbone_type.lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "custom_cnn":
        model = custom_cnn()
        model.out_dim= 1152
        return model.eval()
    elif name == 'clip':
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained= 'openai')
        model.out_dim = 768
        return model.eval()
    elif name == "resnet18":
        model = resnet18(pretrained=pretrained,args=args)
        model.out_dim = 512 
        return model.eval()
    elif name == "resnet34":
        model = resnet34(pretrained=pretrained,args=args)
        model.out_dim = 512 
        return model.eval()
    elif name == "resnet50":
        model = resnet50(pretrained=pretrained,args=args)
        model.out_dim = 2048 
        return model.eval()
    elif name == "resnet101":
        model = resnet101(pretrained=pretrained,args=args)
        model.out_dim = 2048 
        return model.eval()
    elif name =="vgg19":
        model = timm.create_model('vgg19', pretrained=True,num_classes=0 )
        model.out_dim = 4096
        return model.eval()
    elif name =="densenet":
        model = timm.create_model('densenet121', pretrained=True, num_classes=0)
        model.out_dim = 1024
        return model.eval()
    elif name == "vit_tiny_patch16_224":
        from backbone.TinyViT.models.tiny_vit import tiny_vit_5m_224
        model = tiny_vit_5m_224(pretrained=True)
        model.out_dim = 320
        return model.eval()
    #SSF 
    elif '_ssf' in name:
        from backbone import vit_ssf
        if name == "pretrained_vit_b16_224_ssf":
            model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        elif name == "pretrained_vit_b16_224_in21k_ssf":
            model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        elif name == "vit_tiny_patch16_224_ssf":
            model = timm.create_model("vit_tiny_patch16_224_ssf", pretrained=True, num_classes=0)
            model.out_dim = 192
            for name, param in model.named_parameters():
                # Enable training for SSF parameters
                if "ssf" in name:
                    param.requires_grad = True
                elif 'fc.weight' in name or 'fc.sigma' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif name == "vit_tiny_patch16_224_ssf_all" : 
            model = timm.create_model("vit_tiny_patch16_224_ssf", pretrained=True, num_classes=0)
            model.out_dim = 192       
        if args["attention_fusion"] == True:
        ## Assuming 'self.backbone_vit' is your model
            # Enable training for SSF parameters
            if "ssf" in name:
                param.requires_grad = True  
            else:
                param.requires_grad = False
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(args, args["backbone_type"], pretrained)
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        # for RanPAC
        self.W_rand = None
        self.RP_dim = None
        self.backbone_type = args["backbone_type"]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        if "resnet" in self.backbone_type:
            return self.backbone(x)['features']
        elif self.backbone_type == 'clip':
            return self.backbone.encode_image(x)
        else:
            return self.backbone(x)
        
    def forward(self, x):
        x = self.extract_vector(x)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = self.fc(x)
        out.update({"features": x})
        return out


class SimpleVitNet_Attention(nn.Module):
    def __init__(self, args, pretrained):
        super(SimpleVitNet_Attention, self).__init__()
        # for RanPAC
        self.args = args
        self.W_rand = None
        self.RP_dim = None
        self.backbone_type_vit = args["backbone_type_vit"]
        self.backbone_type_cnn = args["backbone_type_cnn"]
        self.backbone_vit = get_backbone(args, self.backbone_type_vit, pretrained= True)
        self.backbone_cnn = get_backbone(args, self.backbone_type_cnn, pretrained= True)
        self.attention_layer = attention_layer(emb_dim=args["d_model"],
        tf_layers= args["tf_layers"], tf_head= args["tf_head"], tf_dim=args["ff_dim"])
        self.fc = None
        self._device = args["device"][0]
        
    @property
    def feature_dim(self):
        return self.args["d_model"]
    
    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x_vit, x_cnn):

        if "resnet" in self.backbone_type_vit:
            vit_features = self.backbone_vit(x_vit)['features']
        elif self.backbone_type_vit == 'clip':
            vit_features= self.backbone_vit.encode_image(x_vit)
        else:
            vit_features= self.backbone_vit(x_vit)

        if "resnet" in self.backbone_type_cnn:
            cnn_features = self.backbone(x_cnn)['features']
        elif self.backbone_type_cnn == 'clip':
            cnn_features= self.backbone_cnn.encode_image(x_cnn)
        else:
            cnn_features= self.backbone_cnn(x_cnn)
        features = self.attention_layer(vit_features, cnn_features)
        return features
    
    def forward(self, x_vit, x_cnn):
        x = self.extract_vector(x_vit, x_cnn)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        #print("x before fc", x.shape)
        out = self.fc(x)
        out.update({"features": x})
        return out