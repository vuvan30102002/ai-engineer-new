from lib import *
from make_datapath import make_data_path_list
from transform import DataTransform
from extract_info_annotation import Anno_xml



class MyDataset(Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        super().__init__()
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)
        return img, gt, height, width
    
    def pull_item(self, index):
        path_img_file = self.img_list[index]
        img_read = cv2.imread(path_img_file)
        height, width, channels = img_read.shape

        path_anno_file = self.anno_list[index]
        anno_info = self.anno_xml(path_anno_file, width, height)
        img, boxes, labels = self.transform(img_read, self.phase, anno_info[:, : 4], anno_info[:, 4])

        # BGR -> RGB
        # (h,w,c) -> (c,h,w)
        img = torch.from_numpy(img[:,:, (2,1,0)]).permute(2,0,1)
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width
    
def my_collate_fn(batch):
    targets = list()
    imgs = list()

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets


if __name__ == "__main__":
    root_path = "/home/quangvux/Bright_Soft_Project/AI Automation/Object-Detection/VOC2012_train_val"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)
    idx = 1
    phase = "train"
    input_size = 300
    color_mean = (104,117,123)
    transform = DataTransform(input_size, color_mean)
    classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
    anno_xml = Anno_xml(classes)
    dataset = MyDataset(train_img_list, train_annotation_list, phase, transform, anno_xml)
    # print(dataset.__getitem__(idx))

    batch_size = 4
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train" : train_dataloader,
        "val" : val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, tartgets = next(batch_iter)

    print(images.size())
    print(tartgets[0].size())



