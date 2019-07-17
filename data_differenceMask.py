

import os, sys, gc, argparse, numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.models import GeneratorCoarse, Discriminator
from datasets.dataloader import data_loader
from utils.utils import ReplayBuffer, weights_init_normal, LambdaLR
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu,threshold_adaptive
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation
from skimage.exposure import rescale_intensity


def get_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataroot", default = "data")
	parser.add_argument("--datamode", default = "train")
	parser.add_argument('--load_path_Shape', type=str,default="/home/np9207/PolyGan/results/Shape/Gan_43.pth", help='load_Shape_model')
	parser.add_argument('--load_path_Stitch', type=str,default="/home/np9207/PolyGan/results/Stitch/Gan_37.pth", help='load_Stitch_model')
	parser.add_argument('--results_path', type=str, default='./', help='save all result folder')



	opt = parser.parse_args()
	return opt
# **** creating difference mask for training data ****

def diffMask(img1,opt,dataset,netG1,netG2):
    res_path = "{}/{}/".format(os.getcwd(),opt.results_path)
    if os.path.isdir("{}/stitched".format(res_path))==False:
            os.mkdir("{}/stitched".format(res_path))
    if os.path.isdir("{}/diffMas".format(res_path))==False:
            os.mkdir("{}/diffMas".format(res_path))
    # netG1 = args[0]
    # netG2 = args[1] 
    
    resize2 = transforms.Resize(size=(128, 128))
    src,mask,style_img,target,gt_cloth,skel,cloth = dataset.get_img("{}_0.jpg".format(img1[:-6]),"{}_1.jpg".format(img1[:-6]))
    src,mask,style_img,target,gt_cloth,skel,cloth = src.unsqueeze(0),mask.unsqueeze(0),style_img.unsqueeze(0),target.unsqueeze(0),gt_cloth.unsqueeze(0),skel.unsqueeze(0),cloth.unsqueeze(0)#, face.unsqueeze(0)
    src1,mask1,style_img1,target1,gt_cloth1,skel1,cloth1 = Variable(src.cuda()),Variable(mask.cuda()),Variable(style_img.cuda()),Variable(target.cuda()),Variable(gt_cloth.cuda()),Variable(skel.cuda()),Variable(cloth.cuda())#, Variable(face.cuda())

    
    gen_targ_Shape,s_128,s_64,s_32,s_16,s_8,s_4 = netG1(skel1,cloth1) # gen_targ11 is structural change cloth
    gen_targ_Stitch,s_128,s_64,s_32,s_16,s_8,s_4 = netG2(src1,gen_targ_Shape,skel1) # gen_targ12 is stitch image
    
    # saving structural 
    pic_Stitch = (torch.cat([gen_targ_Stitch], dim=0).data + 1) / 2.0
#     save_dir = "/home/np9207/PolyGan_res/temp_stitch/"
    
    save_dir2 = "{}/stitched/".format(res_path)
    save_image(pic_Stitch, '{}{}_0.jpg'.format(save_dir2,img1[:-6]), nrow=1)
    #             os.mkdir(save_dir)
    
    save_image(pic_Stitch, './stitch3.jpg', nrow=1)
    
    
    
    msk1 = mask1[0,:,:,:].detach().cpu().permute(1,2,0)
    plt.imsave("./mask3.jpg",msk1,cmap="gray")
    plt.imsave("./ref3.jpg",resize(plt.imread("/home/np9207/vton/data/{}/image/{}_0.jpg".format(opt.datamode,img1[:-6])),(128,128)))
    stitch = rescale_intensity(plt.imread('./stitch3.jpg')/255)
    mask =   rescale_intensity(plt.imread("./mask3.jpg")/255)
    ref = rescale_intensity(plt.imread("./ref3.jpg")/255)
    
    temp_im = ref*(1-mask)
    temp1 = ref*mask # Gives original image without cloth
    temp2 = stitch*mask # Gives 
    temp2[:,:,0][temp2[:,:,0]<0.95]=0
    #     print(lol.shape)

    block_size = 13
    binary = threshold_adaptive(temp2[:,:,0], block_size, offset=0)

    save_diff = "{}/diffMas/".format(res_path)
    plt.imshow(binary*1,cmap="gray")
    plt.imsave("{}{}_0.jpg".format(save_diff,img1[:-6]),binary*1,cmap="gray")



def main():
	opt = get_opt()
	print(opt)
	print("Start to create difference dataset stage:")
	train_loader = data_loader(opt.datamode)

	if not os.path.exists(opt.results_path):
	    os.makedirs(opt.results_path)
	
	
	netG_Shape = GeneratorCoarse(6,3)
	netG_Stitch = GeneratorCoarse(9,3)
	netG_Stitch.cuda()
	netG_Shape.cuda()
	netG_Shape.load_state_dict(torch.load("{}".format(opt.load_path_Shape)))
	netG_Stitch.load_state_dict(torch.load("{}".format(opt.load_path_Stitch)))
	
	
	# create dataset 

	# path to train images of model wearing cloth
	files = os.listdir(opt.image_folder)

	for x in files:
	    diffMask(x,opt,train_loader,netG_Shape,netG_Stitch)


	print('Finished testing %s!')

if __name__ == "__main__":
    main()
