import os, sys, gc, argparse, numpy as np
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.models import GeneratorCoarse, Discriminator
from datasets.dataloader import data_loader
from utils.utils import ReplayBuffer, weights_init_normal, LambdaLR
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu,threshold_adaptive
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation
from skimage.exposure import rescale_intensity
import torchvision.transforms.functional as TF
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 
from PIL import Image

def get_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataroot", default = "data")
	parser.add_argument("--datamode", default = "train")
	parser.add_argument("--stage", default = "Stage1",help='Stage1, Stage2, Stage3')
	parser.add_argument('--Stage1', type=str, default='pre_trained_models/Stage_1/Gan_44.pth', help='load_Stage_1_model')
	parser.add_argument('--Stage2', type=str, default='pre_trained_models/Stage_2/Gan_42.pth', help='load_Stage_2_model')
	parser.add_argument('--Stage3', type=str, default='pre_trained_models/Stage_3/Gan_48.pth', help='load_Stage_3_model')
	parser.add_argument('--results_Stage1', type=str, default='results/test/Stage1', help='save results')
	parser.add_argument('--results_Stage2', type=str, default='results/test/Stage2', help='save results')
	parser.add_argument('--results_Stage3', type=str, default='results/test/Stage3', help='save results')
	parser.add_argument('--model_image', default = "006852_0.jpg",type=str, help='Model of the person wearing cloth')
	parser.add_argument('--reference_image',default= "006852_1.jpg", type=str, help='Reference cloth to swap')

	opt = parser.parse_args()
	return opt

def diffMask(img1=None,img2=None,opt=None,dataset=None,args=None):
	netG = args[0]
	netB = args[1]
	netD = args[2]
	f = args[3]
	res_path = opt.results_Stage3
	res_folders = ['temp_masks',
	 'temp_Stage2',
	 'temp_ref',
	 'temp_diff',
	 'temp_Stage3',
	 'temp_skel',
	 'temp_res',
	 'temp_Stage1',
	 'temp_src']
	for x in res_folders:
	    if os.path.isdir("{}{}".format(res_path,x))==False:
	        os.mkdir("{}{}".format(res_path,x))
	save_masks = "{}{}".format(res_path,"temp_masks")
	save_Stage2 = "{}{}".format(res_path,"temp_Stage2")
	save_ref = "{}{}".format(res_path,"temp_ref")
	save_diff = "{}{}".format(res_path,"temp_diff")
	save_Stage3 = "{}{}".format(res_path,"temp_Stage3")
	save_skel = "{}{}".format(res_path,"temp_skel")
	save_res = "{}{}".format(res_path,"temp_res")
	save_Stage1 = "{}{}".format(res_path,"temp_Stage1")
	save_src = "{}{}".format(res_path,"temp_src")



	resize2 = transforms.Resize(size=(128, 128))
	src,mask,style_img,target,gt_cloth,skel,cloth = dataset.get_img("{}_0.jpg".format(img1[:-6]),"{}_1.jpg".format(img1[:-6]))
	src,mask,style_img,target,gt_cloth,skel,cloth = src.unsqueeze(0),mask.unsqueeze(0),style_img.unsqueeze(0),target.unsqueeze(0),gt_cloth.unsqueeze(0),skel.unsqueeze(0),cloth.unsqueeze(0)#, face.unsqueeze(0)
	src1,mask1,style_img1,target1,gt_cloth1,skel1,cloth1 = Variable(src.cuda()),Variable(mask.cuda()),Variable(style_img.cuda()),Variable(target.cuda()),Variable(gt_cloth.cuda()),Variable(skel.cuda()),Variable(cloth.cuda())#, Variable(face.cuda())
	src,mask,style_img,target,gt_cloth,skel,cloth = dataset.get_img("{}_0.jpg".format(img2[:-6]),"{}_1.jpg".format(img2[:-6]))
	src,mask,style_img,target,gt_cloth,skel,cloth = src.unsqueeze(0),mask.unsqueeze(0),style_img.unsqueeze(0),target.unsqueeze(0),gt_cloth.unsqueeze(0),skel.unsqueeze(0),cloth.unsqueeze(0)#, face.unsqueeze(0)
	src2,mask2,style_img2,target2,gt_cloth2,skel2,cloth2 = Variable(src.cuda()),Variable(mask.cuda()),Variable(style_img.cuda()),Variable(target.cuda()),Variable(gt_cloth.cuda()),Variable(skel.cuda()),Variable(cloth.cuda())



	gen_targ_Stage1,s_128,s_64,s_32,s_16,s_8,s_4 = netG(skel1,cloth2) # gen_targ11 is structural change cloth
	gen_targ_Stage2,s_128,s_64,s_32,s_16,s_8,s_4 = netB(src1,gen_targ_Stage1,skel1) # gen_targ12 is Stage2 image
    
    # saving structural 
	pic_Stage2 = (torch.cat([gen_targ_Stage2], dim=0).data + 1) / 2.0
	#     save_dir = "/home/np9207/PolyGan_res/temp_Stage2/"
	save_image(pic_Stage2, '%s/%d_%s_%d.jpg' % (save_Stage2,f,img1[:-6], 0), nrow=1)

	msk1 = mask1[0,:,:,:].detach().cpu().permute(1,2,0)
	plt.imsave("{}/{}_{}_mask.jpg".format(save_masks,f,img1[:-6]),msk1,cmap="gray")
	plt.imsave("{}/{}_{}_ref.jpg".format(save_ref,f,img1[:-6]),resize(plt.imread("/home/np9207/vton/data/{}/image/{}_0.jpg".format(opt.datamode,img1[:-6])),(128,128)))
	Stage2 = rescale_intensity(plt.imread("{}/{}_{}_0.jpg".format(save_Stage2,f,img1[:-6]))/255)
	mask =   rescale_intensity(plt.imread("{}/{}_{}_mask.jpg".format(save_masks,f,img1[:-6]))/255)
	ref = rescale_intensity(plt.imread("{}/{}_{}_ref.jpg".format(save_ref,f,img1[:-6]))/255)

	temp_im = ref*(1-mask)
	temp1 = ref*mask # Gives original image without cloth
	temp2 = Stage2*mask # Gives 
	temp2[:,:,0][temp2[:,:,0]<0.95]=0
	#     print(lol.Stage1)

	block_size = 13
	binary = threshold_adaptive(temp2[:,:,0], block_size, offset=0)


	plt.imshow(binary*1,cmap="gray")
	plt.imsave("{}/{}_{}_diff.jpg".format(save_diff,f,img1[:-6]),binary*1,cmap="gray")
	diff = plt.imread("{}/{}_{}_diff.jpg".format(save_diff,f,img1[:-6]))
	diff =  Image.fromarray(np.uint8(diff))
	diff = resize2(diff)
	diff = TF.to_tensor(diff)
	diff = TF.normalize(diff,(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
	diff = diff.unsqueeze(0)
	diff = Variable(diff.cuda())
    
	gen_targ_Stage3,s_128,s_64,s_32,s_16,s_8,s_4 = netD(diff,gen_targ_Stage2)

	pic = (torch.cat([gen_targ_Stage3], dim=0).data + 1) / 2.0

	save_image(pic, '{}/{}_{}_{}.jpg'.format(save_Stage3,f, img1[:-6], 54), nrow=1)

	pic2 = (torch.cat([ gen_targ_Stage1], dim=0).data + 1) / 2.0
	pic3 = (torch.cat([ skel1], dim=0).data + 1) / 2.0

	pic00 = (torch.cat([src1], dim=0).data + 1) / 2.0
	save_image(pic00, '{}/{}_{}_src.jpg'.format(save_src,f,img1[:-6]), nrow=3)
	save_image(pic2, '{}/{}_{}_{}_Stage1.jpg'.format(save_Stage1,f, img1[:-6],img2[:-6]), nrow=1)
	save_image(pic3, '{}/{}_{}_skel.jpg'.format(save_skel,f,img1[:-6]), nrow=1)


def saveFullTranslation(image1=None,image2=None,opt=None,f=0):
    
    res_path = opt.results_Stage3
    save_masks = "{}{}".format(res_path,"temp_masks")
    save_Stage2 = "{}{}".format(res_path,"temp_Stage2")
    save_ref = "{}{}".format(res_path,"temp_ref")
    save_diff = "{}{}".format(res_path,"temp_diff")
    save_Stage3 = "{}{}".format(res_path,"temp_Stage3")
    save_skel = "{}{}".format(res_path,"temp_skel")
    save_res = "{}{}".format(res_path,"temp_res")
    save_Stage1 = "{}{}".format(res_path,"temp_Stage1")
    save_src = "{}{}".format(res_path,"temp_src")
    
    
    
    
    
    Stage3_img = rescale_intensity(plt.imread('{}/{}_{}_{}.jpg'.format(save_Stage3,f, image1[:-6], 54))/255)
    diff_img = rescale_intensity(plt.imread("{}/{}_{}_diff.jpg".format(save_diff,f,image1[:-6]))/255)
    Stage2_img = rescale_intensity(plt.imread("{}/{}_{}_0.jpg".format(save_Stage2,f,image1[:-6]))/255)
    img4 = Stage2_img*(1-diff_img)
    img5 = binary_closing(diff_img[:,:,0],)
    ms2 = img5*1
    ms2 = np.expand_dims(ms2,axis=2)
    ms2 = np.repeat(ms2,repeats=3,axis=2)
    img7 = Stage2_img*(1-ms2)
    img8 = Stage3_img*ms2
    im88 = ((img7+img8)-img8.min())/(img8.max()-img8.min())
#     pdb.set_trace()
    ms = rescale_intensity(plt.imread("{}/{}_{}_mask.jpg".format(save_masks,f,image1[:-6]))/255)
    thresh = threshold_otsu(ms[:,:,0])

    binary1 = ms[:,:,0] > thresh

    ms = binary1*1
    ms = np.expand_dims(ms,axis=2)
    ms = np.repeat(ms,repeats=3,axis=2)

    im2 = rescale_intensity(plt.imread("{}/{}_{}_ref.jpg".format(save_ref,f,image1[:-6]))/255)

    im3 = im2*(1-ms)
    im3[im3==0]=1

    plt.imsave("./res_1.jpg",im3*im88)
    res = rescale_intensity(plt.imread("./res_1.jpg")/255)
    plt.imsave("{}/{}_{}.jpg".format(save_res,f,image1[:-6]),res)


def test(opt,test_loader,image1,image2,*args):


	src,mask,style_img,target,gt_cloth,skel,cloth = test_loader.get_img(image1,image2)
	src,mask,style_img,target,gt_cloth,skel,cloth = src.unsqueeze(0),mask.unsqueeze(0),style_img.unsqueeze(0),target.unsqueeze(0),gt_cloth.unsqueeze(0),skel.unsqueeze(0),cloth.unsqueeze(0)
	src,mask,style_img,target,gt_cloth,skel,cloth = Variable(src.cuda()),Variable(mask.cuda()),Variable(style_img.cuda()),Variable(target.cuda()),Variable(gt_cloth.cuda()),Variable(skel.cuda()),Variable(cloth.cuda())
	if opt.stage =="Stage1":
		netG = args[0]
		gen_targ,_,_,_,_,_,_ = netG(skel,cloth) # src,conditions
		pic = (torch.cat([gen_targ], dim=0).data + 1) / 2.0
		save_dir = "{}/{}".format(os.getcwd(),opt.results_Stage1)
		save_image(pic, '{}/{}_{}'.format(save_dir,args[1], opt.model_image), nrow=1)

	elif opt.stage == "Stage2":
		netG1 = args[0]
		netG2 = args[1]
		gen_targ_Stage1,_,_,_,_,_,_ = netG1(skel,cloth)
		gen_targ_Stage2,_,_,_,_,_,_ = netG2(src,gen_targ_Stage1,skel)
		pic1 = (torch.cat([gen_targ_Stage1], dim=0).data + 1) / 2.0
		pic2 = (torch.cat([gen_targ_Stage2], dim=0).data + 1) / 2.0
		save_dir1 = "{}/{}".format(os.getcwd(),opt.results_Stage1)
		save_image(pic1, '{}/{}_{}'.format(save_dir1,args[2], opt.model_image), nrow=1)
		save_dir2 = "{}/{}".format(os.getcwd(),opt.results_Stage2)
		save_image(pic2, '{}/{}_{}'.format(save_dir2,args[2], opt.model_image), nrow=1)

	elif opt.stage == "Stage3":
		diffMask(image1,image2,opt,test_loader,args)
		saveFullTranslation(image1,image2,opt,args[3])



def main():
	opt = get_opt()
	print(opt)
	print("Start to test stage: %s" % (opt.stage))
	    

	# create dataset 
	test_loader = data_loader(opt.datamode)

	    
	if not os.path.exists(opt.results_Stage2):
	    os.makedirs(opt.results_Stage2)
	if not os.path.exists(opt.results_Stage1):
		os.makedirs(opt.results_Stage1)
	if not os.path.exists(opt.results_Stage3):
		os.makedirs(opt.results_Stage3)

	if opt.stage=="Stage1":
		netG_Stage1 = GeneratorCoarse(6,3)
		netG_Stage1.cuda()
		netG_Stage1.load_state_dict(torch.load("{}".format(opt.Stage1)))
		test(opt,test_loader,opt.model_image,opt.reference_image,netG_Stage1,1)
	elif opt.stage == "Stage2":
		netG_Stage1 = GeneratorCoarse(6,3)
		netG_Stage2 = GeneratorCoarse(9,3)
		netG_Stage2.cuda()
		netG_Stage1.cuda()
		netG_Stage1.load_state_dict(torch.load("{}".format(opt.Stage1)))
		netG_Stage2.load_state_dict(torch.load("{}".format(opt.Stage2)))
		test(opt,test_loader,opt.model_image,opt.reference_image,netG_Stage1,netG_Stage2,1)
	elif opt.stage == "Stage3":
		netG_Stage1 = GeneratorCoarse(6,3)
		netG_Stage2 = GeneratorCoarse(9,3)
		netG_Stage3 = GeneratorCoarse(6,3)
		netG_Stage2.cuda()
		netG_Stage1.cuda()
		netG_Stage3.cuda()
		netG_Stage1.load_state_dict(torch.load("{}".format(opt.Stage1)))
		netG_Stage2.load_state_dict(torch.load("{}".format(opt.Stage2)))
		netG_Stage3.load_state_dict(torch.load("{}".format(opt.Stage3)))
		test(opt,test_loader,opt.model_image,opt.reference_image,netG_Stage1,netG_Stage2,netG_Stage3,1)


	print('Finished testing %s!' % (opt.stage))

if __name__ == "__main__":
    main()
