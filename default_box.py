from lib import *

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
class Defbox():
    def __init__(self, cfg):
        self.input_size = cfg["input_size"]
        self.num_classes = cfg["num_classes"]
        self.bbox_aspect_num = cfg["bbox_aspect_num"]
        self.feature_map = cfg["feature_map"]
        self.steps = cfg["steps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
    
    def create_defbox(self):
        defbox_list = list()

        for k, f in enumerate(self.feature_map):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.input_size / self.steps[k]
                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k

                s_k = self.min_size[k]/self.input_size
                defbox_list += [cx, cy, s_k, s_k]

                s_k_ = math.sqrt(s_k * (self.max_size[k] / self.input_size))
                defbox_list += [cx, cy, s_k_, s_k_]

                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    defbox_list += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
            
        output = torch.Tensor(defbox_list).view(-1,4)
        output.clamp_(max=1, min=0)

        return output
    
if __name__ == "__main__":
    defbox = Defbox(cfg)
    defbox_list = defbox.create_defbox()
    print(defbox_list.shape)