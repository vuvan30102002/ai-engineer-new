from lib import *
def make_data_path_list(root_path):
    image_path_template = osp.join(root_path,"JPEGImages","%s.jpg")
    annotation_path_template = osp.join(root_path,"Annotations","%s.xml")

    train_id_names = osp.join(root_path,"ImageSets/Main/train.txt")
    val_id_names = osp.join(root_path,"ImageSets/Main/val.txt")

    train_img_list = list()
    train_annotation_list = list()
    val_img_list = list()
    val_annotation_list = list()

    for line in open(train_id_names):
        id = line.strip()
        train_img_path = (image_path_template % id)
        train_annotation_path = (annotation_path_template % id)
        train_img_list.append(train_img_path)
        train_annotation_list.append(train_annotation_path)

    for line in open(val_id_names):
        id = line.strip()
        val_img_path = (image_path_template % id)
        val_annotation_path = (annotation_path_template % id)
        val_img_list.append(val_img_path)
        val_annotation_list.append(val_annotation_path)
    
    return train_img_list, train_annotation_list, val_img_list, val_annotation_list


if __name__ == "__main__":
    root_path = "/home/quangvux/Bright_Soft_Project/AI Automation/Object-Detection/VOC2012_train_val"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)

    print((train_annotation_list[0]))
