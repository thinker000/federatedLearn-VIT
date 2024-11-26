import os
import argparse
import torch
import torch.nn.functional as F
from torch import optim
from clients import ClientsGroup
from Models import ViT
import numpy as np

parser = argparse.ArgumentParser(description="FedAvg with Vision Transformer")
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID to use')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='Number of clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='Fraction of clients participating in each round')
parser.add_argument('-E', '--epoch', type=int, default=5, help='Local epochs')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='Local batch size')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('-vf', '--val_freq', type=int, default=5, help='Validation frequency')
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='Model save frequency')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='Number of communication rounds')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='Checkpoint save path')
parser.add_argument('-iid', '--IID', type=int, default=0, help='Data distribution IID or non-IID')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    local_weights_path = "./vit_b_16-c867db91.pth"
    net = ViT(num_classes=10, local_weights_path=local_weights_path).to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('cifar10', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = max(int(args['num_of_clients'] * args['cfraction']), 1)

    global_parameters = {key: var.clone() for key, var in net.state_dict().items()}

    for i in range(args['num_comm']):
        print(f"Communication round {i + 1}")

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = [f'client{idx}' for idx in order[:num_in_comm]]

        sum_parameters = None
        for client in clients_in_comm:
            local_parameters = myClients.clients_set[client].localUpdate(
                args['epoch'], args['batchsize'], net, loss_func, opti, global_parameters
            )
            if sum_parameters is None:
                sum_parameters = {key: var.clone() for key, var in local_parameters.items()}
            else:
                for key in sum_parameters:
                    sum_parameters[key] += local_parameters[key]

        global_parameters = {key: var / num_in_comm for key, var in sum_parameters.items()}

        if (i + 1) % args['val_freq'] == 0:
            net.load_state_dict(global_parameters, strict=True)
            total_acc, total = 0, 0
            with torch.no_grad():
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data).argmax(dim=1)
                    total_acc += (preds == label).sum().item()
                    total += label.size(0)
            print(f"Accuracy after round {i + 1}: {total_acc / total:.4f}")

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net.state_dict(), os.path.join(args['save_path'], f"round_{i + 1}.pth"))
