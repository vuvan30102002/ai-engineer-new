from lib import *
from make_datapath import make_data_path_list
from dataset import MyDataset, my_collate_fn
from transform import DataTransform
from extract_info_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#create dataloader
root_path = "/home/quangvux/Bright_Soft_Project/AI Automation/Object-Detection/VOC2012_train_val"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_data_path_list(root_path)
classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
color_mean = [104, 117, 123]
input_size = 300
phase_train = "train"
phase_val = "val"
transform = DataTransform(input_size, color_mean)
anno_xml = Anno_xml(classes)

train_dataset = MyDataset(train_img_list, train_anno_list, phase_train, transform, anno_xml)
val_dataset = MyDataset(val_img_list, val_anno_list, phase_val, transform, anno_xml)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {
    "train" : train_dataloader,
    "val" : val_dataloader
}
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
network = SSD(phase_train, cfg)

vgg_weight = torch.load('./VOC2012_train_val/weights/vgg16_reducedfc.pth')
network.vgg.load_state_dict(vgg_weight)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

network.create_extract.apply(weights_init)
network.create_loc.apply(weights_init)
network.create_conf.apply(weights_init)

# print(network)

criterion = MultiBoxLoss(jaccard_threshold=0.5, negative_pos=3, device=device)
optimizer = optim.SGD(network.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

def train_model(network, dataloader_dict, criterion, optimizer, num_epochs):
    network.to(device)
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("---"*20)
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("---"*20)

        for phase in ["train", "val"]:
            if phase == "train":
                network.train()
                print("(Training)")
            else:
                if (epoch+1) % 10 == 0:
                    network.val()
                    print("---"*10)
                    print("(Validation)")
                else:
                    continue
            for images, targets in dataloader_dict[phase]:
                images = images.to(device)
                

                targets = [ann.to(device) for ann in targets]

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = network(images)

                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_value_(network.parameters(), clip_value=2.0)
                        optimizer.step()
                        if iteration % 10 == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print("Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec".format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
            
        t_epoch_end = time.time()
        print("---"*20)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f}".format(epoch + 1, epoch_train_loss, epoch_val_loss))
        print("Duration: {:.4f} sec".format(t_epoch_end - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {
            "epoch" : epoch + 1,
            "train_loss" : epoch_train_loss,
            "val_loss" : epoch_val_loss
        }
        logs.append(log_epoch)


        with open("./train_log.json", "w") as f:
            json.dump(logs, f, indent=2)

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        if ((epoch + 1) % 10 == 0):
            torch.save(network.state_dict(), "./VOC2012_train_val/weights/ssd300" + str(epoch + 1) + ".pth")

num_epoch = 30
train_model(network, dataloader_dict, criterion, optimizer, num_epoch)
