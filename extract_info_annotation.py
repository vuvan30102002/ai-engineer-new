from lib import *
from make_datapath import make_data_path_list

class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes
        ret = list()
        
    def __call__(self, xml_path, width, height):
        object = ET.parse(xml_path).getroot()
        for obj in object.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            name = obj.find('name').text.strip().lower()
            bbox = list()
            box = ["xmin","ymin","xmax","ymax"]
            bndbox = obj.find("bndbox")
            for bb in box:
                pixel = int(bndbox.find(bb).text) - 1
                if bb == "xmin" or bb == "xmax":
                    pixel = pixel / width
                else:
                    pixel = pixel / height
                bbox.append(pixel)
            bbox.append(self.classes.index(name))
        ret += [bbox]
        return np.array(ret)

if __name__ == "__main__":
    path = ""
    classes = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    anno_xml = Anno_xml(classes)
    root_path = "/home/quangvux/Bright_Soft_Project/AI Automation/Object-Detection/VOC2012_train_val"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)
    idx = 1
    xml_path = train_annotation_list[idx]
    img_path = train_img_list[idx]
    read_img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    height, width, channel = read_img.size()
    bbox_info = anno_xml(xml_path, width, height)
    print(bbox_info)