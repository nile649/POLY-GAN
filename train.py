import os, sys, gc, argparse, numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.models import GeneratorCoarse, Discriminator
from datasets.dataloader import PolyDatasetShape, PolyDatasetStitch
from utils.utils import ReplayBuffer, weights_init_normal, LambdaLR


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--j', type=int, default=0)
    parser.add_argument('--b', type=int, default=1)
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "Shape",help='Shape, Stitch, Refine')
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument("--display_count", type=int, default = 1000)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--epochs", type=int, default = 45)
    parser.add_argument("--input_channel", type=int, default = 6)
    parser.add_argument("--decay_epoch", type=int, default = 10)
    parser.add_argument('--results', type=str, default='results/Shape', help='save results')
    parser.add_argument("--critic", type=int, default = 10)
    parser.add_argument("--save_model", type=int, default = 2)
    opt = parser.parse_args()
    return opt

def train(opt,train_loader,netG,netD):
    epoch = 0
    n_epochs = opt.epochs
    decay_epoch = opt.decay_epoch
    batchSize = opt.b
    size = 128
    input_nc = opt.input_channel
    output_nc = 3
    lr = opt.lr
    if opt.stage!="Refine":
        nRow = 3
    else:
        nRow = 4
    
    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(),lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(batchSize, input_nc, size, size)
    target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

    fake_buffer = ReplayBuffer()
    
    for epoch in range(0, n_epochs):
        gc.collect()
        Source = iter(train_loader)
        avg_loss_g = 0
        avg_loss_d = 0
        for i in range(0,len(train_loader)):
            netG.train()
            target_real = Variable(torch.ones(1,1), requires_grad=False).cuda()
            target_fake = Variable(torch.zeros(1,1), requires_grad=False).cuda()
            optimizer_G.zero_grad()

            if opt.stage!="Refine":
                src,mask,style_img,target,gt_cloth,skel,cloth = Source.next()
                src,mask,style_img,target,gt_cloth,skel,cloth = Variable(src.cuda()),Variable(mask.cuda()),Variable(style_img.cuda()),Variable(target.cuda()),Variable(gt_cloth.cuda()),Variable(skel.cuda()),Variable(cloth.cuda())
            else:
                src,mask,style_img,target,gt_cloth,wrap,diff,cloth = Source.next()
                src,mask,style_img,target,gt_cloth,wrap,diff,cloth = Variable(src.cuda()),Variable(mask.cuda()),Variable(style_img.cuda()),Variable(target.cuda()),Variable(gt_cloth.cuda()),Variable(wrap.cuda()),Variable(diff.cuda()),Variable(cloth.cuda())

            #Inverse identity
            if opt.stage=="Shape":
                gen_targ,_,_,_,_,_,_ = netG(skel,cloth) # src,conditions
            elif opt.stage == "Stitch":
                gen_targ,_,_,_,_,_,_ = netG(src,style_img,skel)
            elif opt.stage == "Refine":
                gen_targ,_,_,_,_,_,_ = netG(diff,wrap)
                
            pred_fake = netD(gen_targ)
            
            if opt.stage=="Shape":
                loss_GAN = 10*criterion_GAN(pred_fake, target_real) + 10*criterion_identity(gen_targ, gt_cloth)
            elif opt.stage == "Stitch" or opt.stage == "Refine":
                loss_GAN = 10*criterion_GAN(pred_fake, target_real) + 10*criterion_identity(gen_targ, target)

            loss_G = loss_GAN
            loss_G.backward()

            optimizer_G.step()        
            #############################################

            optimizer_D.zero_grad()

            if opt.stage=="Shape":
                pred_real = netD(gt_cloth)
            elif opt.stage == "Stitch" or opt.stage == "Refine":
                pred_real = netD(target)
            
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            gen_targ = fake_buffer.push_and_pop(gen_targ)
            pred_fake = netD(gen_targ.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D = (loss_D_real + loss_D_fake)*0.5
            loss_D.backward()
            if (i + 1) % opt.critic == 0:
                optimizer_D.step()

            avg_loss_g = (avg_loss_g+loss_G)/(i+1) 
            avg_loss_d = (avg_loss_d+loss_D)/(i+1) 

            if (i + 1) % 100 == 0:
                print("Epoch: (%3d) (%5d/%5d) Loss: (%0.0003f) (%0.0003f)" % (epoch, i + 1, len(train_loader), avg_loss_g*1000, avg_loss_d*1000))


            if (i + 1) % opt.display_count == 0:  
                if opt.stage=="Shape":
                    pic = (torch.cat([style_img, gen_targ, cloth,skel, target,gt_cloth], dim=0).data + 1) / 2.0
                elif opt.stage=="Stitch":
                    pic = (torch.cat([src, gen_targ, cloth,skel, target,gt_cloth], dim=0).data + 1) / 2.0
                elif opt.stage=="Refine":
                    pic = (torch.cat([wrap,diff,gen_targ, target], dim=0).data + 1) / 2.0

                save_dir = "{}/{}".format(os.getcwd(),opt.results)
    #             os.mkdir(save_dir)
                save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(train_loader)), nrow=nRow)
        if (epoch + 1) % opt.save_model == 0:
            save_dir = "{}/{}".format(os.getcwd(),opt.results)
            torch.save(netG.state_dict(), '{}/Gan_{}.pth'.format(save_dir,epoch))
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s" % (opt.stage))
        
    
    # create dataset 
    if opt.stage=="Shape":
        dataset = PolyDatasetShape(128)
        train_loader = DataLoader(dataset,
                         batch_size=opt.b,
                         shuffle=False,
                         num_workers=opt.j,
                         drop_last=True,pin_memory=True)
        
        
    elif opt.stage=="Stitch":
        dataset = PolyDatasetStitch(128)
        train_loader = DataLoader(dataset,
                         batch_size=opt.b,
                         shuffle=False,
                         num_workers=opt.j,
                         drop_last=True,pin_memory=True)

        
    elif opt.stage=="Refine":
        dataset = PolyDatasetRefine(128)
        train_loader = DataLoader(dataset,
                         batch_size=opt.b,
                         shuffle=False,
                         num_workers=opt.j,
                         drop_last=True,pin_memory=True)
    else:
        sys.exit("Please mention the Stage from [Shape, Stitch, Refine]")

        
    if not os.path.exists(opt.results):
        os.makedirs(opt.results)
    netG = GeneratorCoarse(opt.input_channel,3)
    netD = Discriminator()   
    # create model & train & save the final checkpoint
    netG.cuda()

    netD.cuda()


    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)
        
    train(opt,train_loader,netG,netD)
    
    print('Finished training %s!' % (opt.stage))

if __name__ == "__main__":
    main()
