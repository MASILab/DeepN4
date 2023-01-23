import torch
from torch.utils.data import DataLoader
from data import *
from batch_logspace_bias import train, test, predict
# from simple_model import *
from model_linear import *
from utils import *
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def pred_model(val_file, checkpoint_file, model_dir,file_dir):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = UNet3D(1, 1).to(device)
    # model = _NetG().to(device)
    model = load_model(model, checkpoint_file)

    with open(val_file, 'r') as f:
        for l in f.readlines():
            paths = l.strip().split(',')
            # outpath = '{}/{}_{}'.format(Path(paths[1]).parent, 'pred', Path(paths[1]).name)
            outpath = os.path.join(file_dir ,'pred_3dunet_img2bias_log.nii.gz')
            outpath_bias = os.path.join(file_dir ,'pred_biasfield_img2bias_log.nii.gz')
            estpath_bias = os.path.join(file_dir ,'est_biasfield_img2bias_log.nii.gz')
            print(outpath)
            # val_dataset = dataset_predict(paths)
            val_dataset = dataset_predict(paths)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)
            predict(model, val_loader, device, paths[1], outpath, outpath_bias, estpath_bias)
            break


def test_model(val_file, checkpoint_file):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    val_dataset = dataset(val_file)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    model = AutoEncoder().to(device=1)

    model = load_model(model, checkpoint_file)

    test_loss = test(model, val_loader, device)#, save_output=True)


def train_model(train_file, val_file, model_dir, ten_out_dir, cont_train=False, checkpoint_file=''):
    out_dir = 'output'
    writer = SummaryWriter(out_dir)
    use_cuda = torch.cuda.is_available()
    print("Running on {}".format(use_cuda))
    torch.manual_seed(1)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = dataset(train_file)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, **kwargs)


    # import pdb;pdb.set_trace()

    val_dataset = dataset(val_file)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    lr = 1e-2
    # lr = 0.0001
    # import pdb; pdb.set_trace()
    model = AutoEncoder().to(device)
    torch.cuda.empty_cache()
    # model = _NetG().to(device)
    # model = Duelling_CNN().to(device)
    # duel = Duelling_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    epochs = 100

    min_loss = 9999999999

    if cont_train:
        start_epoch = int(checkpoint_file.split('_')[-1])+1
        model, optimizer = load_checkpoint(model, optimizer, checkpoint_file)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if not cont_train:
        f = open('{}/train_loss.txt'.format(model_dir), 'w')
        f.close()
        f = open('{}/val_loss.txt'.format(model_dir), 'w')
        f.close()

    for epoch in range(start_epoch, epochs+1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        val_loss = test(model, val_loader, device)

        with open('{}/train_loss.txt'.format(model_dir), 'a') as f:
            f.write('{}\n'.format(train_loss))
        with open('{}/val_loss.txt'.format(model_dir), 'a') as f:
            f.write('{}\n'.format(val_loss))

        if val_loss < min_loss:
            min_loss = val_loss
            save_model(model, optimizer, '{}/checkpoint_epoch_{}'.format(model_dir, epoch))

        writer.add_scalars('Loss',   {'Train_3dunet_ae': train_loss, 'Validation_3dunet_ae': val_loss}, epoch)

    
        # if epoch%10 == 0:
        #     lr = lr/2
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr


def main():
    # test_file = '../folds/test.csv'
    train_file = '/nfs/masi/kanakap/projects/DeepN4/src/train_5ds.csv'
    val_file = '/nfs/masi/kanakap/projects/DeepN4/src/val_5ds.csv'
    test_file = '/nfs/masi/kanakap/projects/DeepN4/src/all_test.csv'
    #test_file = '/nfs/masi/kanakap/projects/DeepN4/src/test_train_5ds.csv'

    model_dir = '/nfs/masi/kanakap/projects/DeepN4/src/unet_trained_model_logsub2'
    fcheckpoint = 'checkpoint_epoch_99'
    file_dir = '/nfs/masi/kanakap/projects/DeepN4/src/unet_trained_model_all_noguass'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    ten_out_dir = '/nfs/masi/kanakap/projects/DeepN4/src/output2'

    #train_model(train_file, val_file, model_dir, ten_out_dir, cont_train=False, checkpoint_file='{}/{}'.format(model_dir, fcheckpoint))
    # test_model(test_file, '{}/{}'.format(model_dir, fcheckpoint))
    pred_model(test_file, '{}/{}'.format(model_dir, fcheckpoint),model_dir,file_dir)

if __name__ == '__main__':
    main()
