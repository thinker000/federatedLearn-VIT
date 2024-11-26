import torch
from torch.utils.data import DataLoader, Subset
from getData import GetDataSet

class client(object):
    def __init__(self, trainDataSet, dev):
        """初始化客户端"""
        self.train_ds = trainDataSet
        self.dev = dev

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        """客户端本地模型更新"""
        Net.load_state_dict(global_parameters, strict=True)
        train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()


class ClientsGroup:
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.clients_set = {}
        self.test_data_loader = None

        # 加载数据集
        dataset = GetDataSet(dataSetName, isIID)

        # 获取测试数据加载器
        self.test_data_loader = dataset.getTestLoader(batch_size=100)

        # 划分训练数据到每个客户端
        shard_size = len(dataset.train_set) // numOfClients
        for i in range(numOfClients):
            data_shard = Subset(dataset.train_set, range(i * shard_size, (i + 1) * shard_size))
            self.clients_set[f'client{i}'] = client(data_shard, dev)
