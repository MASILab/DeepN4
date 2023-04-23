import os
import sys
import torch
from utils import *
from model_all import *
from data_targbias_final import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from batch_targbias_noaug import train, test, predict


def pred_model(model_type, gpu_no, val_file, checkpoint_file, pred_dir, filter_type):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda:" + gpu_no if use_cuda else "cpu")
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

    
    model = model_type(1, 1).to(device)
    model = load_model(model, checkpoint_file)

    with open(val_file, 'r') as f:
        for l in f.readlines():
            paths = l.strip().split(',')
            #paths = l.replace('\n','').split(',')
            # filename = paths[0].split('/')[4].split(',')[0] # test_all_more
            # filename = paths[0].split('/')[7].split(',')[0] # for test_all [vmap]
            filename = paths[0].split('/')[7].split(',')[0] + '_'+ paths[0].split('/')[8].split(',')[0] # for external more
            # outpath = '{}/{}_{}'.format(Path(paths[1]).parent, 'pred', Path(paths[1]).name)
            outpath = os.path.join(pred_dir ,'pred' + filename )
            outpath_bias = os.path.join(pred_dir ,'pred_biasfield' + filename )
            estpath_bias = os.path.join(pred_dir ,'est_biasfield' + filename )
            print(outpath)
            # val_dataset = dataset_predict(paths)
            print(paths)
            val_dataset = dataset_predict(paths)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)
            predict(model, val_loader, device, paths[1], outpath, outpath_bias, estpath_bias, filter_type)


def test_model(model_type, gpu_no, val_file, checkpoint_file):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda:" + gpu_no if use_cuda else "cpu")
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

    val_dataset = dataset(val_file)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    model = model_type(1, 1).to(device)
    model = load_model(model, checkpoint_file)
    test_loss = test(model, val_loader, device)


def train_model(model_type, model_folder, gpu_no, aug, train_file, val_file, model_dir, tb_dir, cont_train=False, checkpoint_file=''):
    writer = SummaryWriter(tb_dir)
    use_cuda = torch.cuda.is_available()
    print("Running training".format(use_cuda))
    torch.manual_seed(1)
    device = torch.device("cuda:" + gpu_no if use_cuda else "cpu")
    print(torch.cuda.get_device_name(int(gpu_no)))
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {} 

    train_dataset = dataset(train_file,transform=aug)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    # import pdb;pdb.set_trace()

    val_dataset = dataset(val_file)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, **kwargs)

    lr = 1e-4
    # lr = 0.0001
    model = model_type(1, 1).to(device)
    # model = _NetG().to(device)
    # model = Duelling_CNN().to(device)
    # duel = Duelling_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    epochs = 1000

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
        train_loss = train(model, train_loader, optimizer, device, epoch, model_dir)
        val_loss = test(model, val_loader, device)

        with open('{}/train_loss.txt'.format(model_dir), 'a') as f:
            f.write('{}\n'.format(train_loss))
        with open('{}/val_loss.txt'.format(model_dir), 'a') as f:
            f.write('{}\n'.format(val_loss))

        # save model for epoch when val loss is less than train loss 
        if val_loss < min_loss:
            min_loss = val_loss
            save_model(model, optimizer, '{}/checkpoint_epoch_{}'.format(model_dir, epoch))
        #change
        writer.add_scalars('Loss',   {'Train_' + model_folder: train_loss, 'Validation_' + model_folder: val_loss}, epoch)


def main():
    # test_file = '../folds/test.csv'
    train_file = '/nfs/masi/kanakap/projects/DeepN4/src/train_more_all.csv'
    val_file = '/nfs/masi/kanakap/projects/DeepN4/src/val_more_all.csv'
    test_file = '/nfs/masi/kanakap/projects/DeepN4/src/test_more_external1.csv' # test_all [vmap] # test_more_external1
    #test_file = '/nfs/masi/kanakap/projects/DeepN4/src/cbtest_train_5ds.csv'

    run_type = sys.argv[1]         # train / test/ pred
    gpu_no = sys.argv[2]           # 0 , 1
    model_type = eval(sys.argv[3]) # Synbo_UNet3D, trad_UNet3D, bspline_UNet3D
    aug = sys.argv[4]              # True / False
    model_folder = sys.argv[5]     # trained_model_<bspline_UNet3D_aug>
    tb_folder = sys.argv[6]        # tensorboad output 
    output_folder = sys.argv[7]    # pred_output_<bspline_UNet3D_aug>
    fcheckpoint = sys.argv[8]      #'checkpoint_epoch_335'
    filter_type = sys.argv[9]      # bspline or guass

    
    model_dir = '/nfs/masi/kanakap/projects/DeepN4/src/trained_model_' + model_folder  
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    tb_dir = '/nfs/masi/kanakap/projects/DeepN4/src/' + tb_folder
    if not os.path.isdir(tb_folder):
        os.mkdir(tb_folder)
    pred_dir = '/nfs/masi/kanakap/projects/DeepN4/src/pred_output_' + output_folder
    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)

    if run_type == 'train':
        train_model(model_type, model_folder, gpu_no, aug, train_file, val_file, model_dir, tb_dir, cont_train=False, checkpoint_file='{}/{}'.format(model_dir, fcheckpoint))
    elif run_type == 'test':
        test_model(test_file,gpu_no, '{}/{}'.format(model_dir, fcheckpoint))
    elif run_type == 'pred': 
        pred_model(model_type, gpu_no, test_file, '{}/{}'.format(model_dir, fcheckpoint),pred_dir, filter_type)

if __name__ == '__main__':
    main()

