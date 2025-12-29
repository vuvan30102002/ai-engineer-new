from lib import *
from l2_norm import L2Norm
from default_box import Defbox

def create_vgg():
    layers = list()
    in_channels = 3
    cfgs = [64,64,"M",128,128,"M",256,256,256,"MC",512,512,512,"M",512,512,512]
    for cfg in cfgs:
        if cfg == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == "MC":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg
        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def create_extract():
    layers = list()
    in_channels = 1024
    cfgs = [256,512,128,256,128,256,128,256]
    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3, stride=2, padding=1)]

    return nn.ModuleList(layers)


def create_loc_conf(num_classes=21, bbox_ratio_num=[4,6,6,6,4,4]):
    loc_layers = list()
    conf_layers = list()

    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        super().__init__()

        self.vgg = create_vgg()
        self.create_extract = create_extract()
        self.create_loc, self.create_conf = create_loc_conf()
        self.L2norm = L2Norm()

        dbox = Defbox(cfg)
        self.dbox_list = dbox.create_defbox()

        if phase == "inference":
            self.detect = Detect()
        

if __name__ == "__main__":
    # vgg = create_loc_conf()
    # print(vgg)
    cfg = {
        "num_classes" : 21,
        "input_size" : 300,
        "bbox_aspect_num": [4,6,6,6,4,4],
        "feature_map" : [38,19,10,5,3,1],
        "steps" : [8,16,32,64,100,300],
        "min_size" : [30,60,111,162,213,264],
        "max_size" : [60,111,162,213,264,315],
        "aspect_ratios" : [[2],[2,3],[2,3],[2,3],[2],[2]]
    }
    ssd = SSD(phase="train", cfg=cfg)

    print(ssd)