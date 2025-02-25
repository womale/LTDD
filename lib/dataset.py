from torch_geometric.data import Dataset
import numpy as np

class graphDataset(Dataset):
    def __init__(self, data_list , args, mode='train'):
        super(graphDataset, self ).__init__()
        self.data_list = data_list
        self.mode=mode
        self.new_labels = []
        self.args=args


        if (args.resample==True) and (self.mode=='train'):
            resample_data_list=[]
            length=len(self.data_list)
            num_classes=args.num_classes
            length=length-length%num_classes
            cls_num_list = [0]*self.args.num_classes
            for i in range(len(self.data_list)):
                cls_num_list[self.data_list[i].y.item()]+=1
            pos=0
            for i in range(num_classes):
                sample_index=np.random.randint(pos, pos+cls_num_list[i], length//num_classes)
                for index in sample_index:
                    resample_data_list.append(self.data_list[index])
                pos+=cls_num_list[i]
            self.data_list=resample_data_list

    def len(self):
        return len(self.data_list)
    
    def get_cls_num_list(self):
        cls_num_list = [0]*self.args.num_classes
        for i in range(len(self.data_list)):
            cls_num_list[self.data_list[i].y.item()]+=1
        return cls_num_list

    def get(self, idx):

        graph=self.data_list[idx]

        if self.new_labels !=[]:
            new_labels = self.new_labels[idx]
        else: 
            new_labels =-1

        return graph,new_labels