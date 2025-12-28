from utils.augmentation import *

class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train" : Compose([
                ConvertFromInts(), #convert image from int to float32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), #change color by random
                Expand(color_mean),
                RandomSampleCrop(),   #random crop image
                RandomMirror(), # 
                ToPercentCoords(),  # normalization data to 0-1
                Resize(),  # resize to 300x300
                SubtractMeans(color_mean),  # subtract mean of BGR
            ]),
            "val" : Compose([
                ConvertFromInts(),
                Resize(),
                SubtractMeans(color_mean)
            ]),
        }
    
    def __call__(self, image, phase, boxes, labels):
        return self.data_transform[phase](image, boxes, labels)