import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from math import log10
import random
import argparse
import datetime
import albumentations as A
from model_2D import *
from utility_2D import *

date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.cuda.empty_cache()

# For multi-gpu
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

Encoder = Encoder().cuda()
IEM = IEM().cuda()
Age_predictor = Age_predictor().cuda()
Generator = Generator().cuda()
Discriminator = Discriminator().cuda()
MappingNetwork = MappingNetwork().cuda()



"""
Load dataset
"""

train_csv = pd.read_csv('../train.csv')
train_npy = np.load("../train.npy", mmap_mode="r")

test_csv = pd.read_csv('/DataCommon2/mjy/data/UK_Biobank/second_visit_final_982.csv')
test_1st_npy = np.load("/DataCommon2/mjy/data/UK_Biobank/2D_image_npy/2mm/Test_first_visit_982_minmax.npy", mmap_mode="r")
test_2nd_npy = np.load("/DataCommon2/mjy/data/UK_Biobank/2D_image_npy/2mm/Test_second_visit_982_minmax.npy", mmap_mode="r")


def main():

    Rc_loss = L2Reconloss().cuda()
    Cycle_loss = nn.L1Loss().cuda()

    best_models = {}

    ####### Optimizers

    optimizer_A = torch.optim.Adam(list(Encoder.parameters()) + list(Age_predictor.parameters()), lr=args.lr_age)
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=args.lr_gan)
    optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=args.lr_gan)
    optimizer_M = torch.optim.Adam(MappingNetwork.parameters(), lr=args.lr_map)
    optimizer_I = torch.optim.Adam(IEM.parameters(), lr=args.lr_id)


    # Scheduler
    if args.scheduler == 'no':
        pass
    elif args.scheduler == 'yes':
        # StepLR
        scheduler_A = torch.optim.lr_scheduler.StepLR(optimizer_A, step_size=30, gamma=0.3)

    ####### Training
    Encoder.train()
    IEM.train()
    Age_predictor.train()
    Generator.train()
    Discriminator.train()
    MappingNetwork.train()

    batch_size, epoch = args.batch_size, args.epoch

    # Attributes for early stopping
    min_mae = np.inf
    not_improve = 0


    for epoch in range(epoch+1):
        print('Epoch {}/{}'.format(epoch, args.epoch))

        D_loss = 0
        G_loss = 0
        A_loss = 0
        N_loss = 0
        I_loss = 0
        # P_loss = 0
        MAE_real = 0
        MAE_inter = 0
        MAE_final = 0

        # utilities for SFCN
        bin_range = [48, 81]
        bin_step = 1
        sigma = 1


        total_idx = np.random.permutation(len(train_npy))
        total_idx2 = np.random.permutation(len(train_npy))

        # Batch loop
        train_loop = len(total_idx)


        for idx in tqdm(range(0, train_loop, batch_size)):
            index1 = total_idx[idx:idx+batch_size]
            index2 = total_idx2[idx:idx+batch_size]
            # load dataset
            image1 = train_npy[index1]
            b, w, d = image1.shape
            image1 = torch.from_numpy(image1).float().view(b, 1, w, d)
            image2 = train_npy[index2]
            image2 = torch.from_numpy(image2).float().view(b, 1, w, d)


            if random.random() <= 0.2:
                torch_aug = torch.nn.Sequential(T.RandomHorizontalFlip(p=0.5), T.GaussianBlur(kernel_size=3))
            else:
                torch_aug = T.RandomHorizontalFlip(p=0.5)
            img_aug = torch_aug(image1)
            age1_ = torch.tensor(train_csv['first_age_condition'].iloc[index1].values)
            age2_ = torch.tensor(train_csv['first_age_condition'].iloc[index2].values)

            image1 = Variable(image1).cuda()
            image2 = Variable(image2).cuda()
            img_aug = Variable(img_aug).cuda()
            age1 = Variable(age1_).long().cuda()
            age2 = Variable(age2_).long().cuda()

            input_age = age1
            input_age = input_age.cuda()

            target_age = age2
            target_age = target_age.cuda()

            age_y_1, bc_1 = num2vect(age1_, bin_range, bin_step, sigma)
            age_y_1 = torch.tensor(age_y_1, dtype=torch.float32).cuda()

            age_y_2, bc_2 = num2vect(age2_, bin_range, bin_step, sigma)
            age_y_2 = torch.tensor(age_y_2, dtype=torch.float32).cuda()

            # for making weights
            diff_age = (age2 - age1) / 33


            """
            Train Age Predictor
            """
            if not_improve < 20:
                optimizer_A.zero_grad()

                Age_predictor_clone = copy.deepcopy(Age_predictor).cuda()

                for param in Age_predictor_clone.parameters():
                    param.requires_grad = False

                in_f, c4, c3, c2, c1 = Encoder(img_aug.float())
                fake_pred_age = Age_predictor(in_f)

                # for first age prediction loss
                # Age Predictor loss
                age_x = fake_pred_age[0].reshape([batch_size, -1])
                loss_A = my_KLDivLoss(age_x.cuda(), age_y_1).cuda()

                # calculate MAE
                x_real = age_x.cpu().detach().numpy()
                prob_input = np.exp(x_real)
                pred_input = prob_input @ bc_1
                pred_input = torch.tensor(pred_input)
                mae_input = torch.abs(pred_input.cuda() - age1).sum() / batch_size

                Age_loss = loss_A
                Age_loss.backward()
                optimizer_A.step()


            """
            Train Discriminator
            """
            if epoch > 0:
                # Load the best model
                Encoder.load_state_dict(best_models['Encoder'])
                Age_predictor.load_state_dict(best_models['Age_predictor'])


            optimizer_D.zero_grad()

            style = MappingNetwork(target_age)
            in_f, c4, c3, c2, c1 = Encoder(image1.float())
            in_f, c4, c3, c2, c1, _, _ = IEM(in_f, c4, c3, c2, c1)
            mask = Generator(in_f, c4, c3, c2, c1, style)
            fake_MRI = mask + image1.float()

            # Real Image
            pred_real = Discriminator(image2.float(), style.detach())

            # Fake Image
            pred_fake = Discriminator(fake_MRI.detach(), style.detach())

            # Discriminator Loss
            loss_D = 0.5 * (torch.mean((pred_real - 1) ** 2) + torch.mean(pred_fake ** 2))

            loss_D.backward()
            optimizer_D.step()


            """
            Train Generator
            """
            optimizer_G.zero_grad()
            optimizer_M.zero_grad()

            style = MappingNetwork(target_age)
            in_f, c4, c3, c2, c1 = Encoder(image1.float())
            in_f, c4, c3, c2, c1, _, _ = IEM(in_f, c4, c3, c2, c1)
            mask = Generator(in_f, c4, c3, c2, c1, style)
            fake_MRI = mask + image1.float()

            pred_real2 = Discriminator(fake_MRI, style.detach())
            loss_G_real = 0.5 * torch.mean((pred_real2 - 1) ** 2)


            in_f_fake_clone, c4, c3, c2, c1 = Encoder(fake_MRI)
            final_pred_age = Age_predictor(in_f_fake_clone)

            ## Age Prediction Loss
            age_x = final_pred_age[0].reshape([batch_size, -1])
            loss_A_final = my_KLDivLoss(age_x.cuda(), age_y_2).cuda()

            # calculate MAE
            x_real = age_x.cpu().detach().numpy()
            prob_input = np.exp(x_real)
            pred_input = prob_input @ bc_2
            pred_input = torch.tensor(pred_input)
            mae_final = torch.abs(pred_input.cuda() - age2).sum() / batch_size

            # L2 Recon Loss
            weights = compute_cosine_weights(diff_age)
            loss_rc = Rc_loss(fake_MRI.cuda(), image1.float(), weights)

            # Cycle Consistency Loss
            style_input = MappingNetwork(input_age)
            in_f_fake_clone, c4, c3, c2, c1, _, _ = IEM(in_f_fake_clone, c4, c3, c2, c1)
            mask = Generator(in_f_fake_clone.detach(), c4.detach(), c3.detach(), c2.detach(), c1.detach(), style_input)
            """
            In Cycle Consistency Loss, we residually add the input image instead of the fake image
            """
            fake_Recon = mask + image1.float()

            loss_cycle = Cycle_loss(image1.float(), fake_Recon)


            loss_G = loss_G_real + 0.05 * loss_A_final + 0.2 * loss_rc + 0.2 * loss_cycle
            loss_G.backward()
            optimizer_G.step()
            optimizer_M.step()


            """
            Train Identity Extractor
            """
            optimizer_I.zero_grad()

            i1, i2, i3, i4, i5 = Encoder(image1.float())
            f1, f2, f3, f4, f5 = Encoder(fake_MRI.detach())

            _, _, _, _, _, p1, z1 = IEM(i1, i2, i3, i4, i5)
            _, _, _, _, _, p2, z2 = IEM(f1, f2, f3, f4, f5)

            # Identity extracting Loss
            loss_I = D(p1, z2) / 2 + D(p2, z1) / 2 + D_orth_2D(i1, i2, i3, i4, i5, z1) / 2 + D_orth_2D(f1, f2, f3, f4, f5, z2) / 2

            loss_I.backward()
            optimizer_I.step()



            A_loss += loss_A.item()
            I_loss += loss_I.item()
            MAE_real += mae_input.item()
            D_loss += loss_D.item()
            G_loss += loss_G_real.item()
            MAE_final += mae_final.item()


        if args.scheduler == 'no':
            pass
        elif args.scheduler == 'yes':
            if not_improve < 20:
                scheduler_A.step()
            else:
                pass


        """
        Quantitative Evaluation Metric
        1. MSE
        2. PSNR
        3. SSIM
        4. PAD
        """

        MSE_pro = 0
        MSE_reg = 0
        PSNR_pro = 0
        PSNR_reg = 0
        SSIM_pro = 0
        SSIM_reg = 0
        MAE_1st = 0
        MAE_2nd = 0
        PAD = 0

        Encoder.eval()
        IEM.eval()
        Generator.eval()
        Age_predictor.eval()
        MappingNetwork.eval()

        # do not permutate in validation loop
        total_val_idx = np.arange(len(test_1st_npy))

        if args.batch_size == 16:
            val_loop = len(total_val_idx) - 6
        if args.batch_size == 32:
            val_loop = len(total_val_idx) - 22
        if args.batch_size == 64:
            val_loop = len(total_val_idx) - 22
        if args.batch_size == 128:
            val_loop = len(total_val_idx) - 86

        # Validation loop
        for idx in tqdm(range(0, val_loop, batch_size)):
            index = total_val_idx[idx:idx + batch_size]
            # load dataset
            first_img_ = test_1st_npy[index]
            second_img_ = test_2nd_npy[index]
            first_age_ = torch.tensor(test_csv['first_age_condition'].iloc[index].values)
            second_age_ = torch.tensor(test_csv['second_age_condition'].iloc[index].values)

            b, w, d = first_img_.shape
            first_img_ = torch.from_numpy(first_img_).float().view(b, 1, w, d)
            second_img_ = torch.from_numpy(second_img_).float().view(b, 1, w, d)

            first_img = Variable(first_img_).cuda()
            second_img = Variable(second_img_).cuda()
            first_age = Variable(first_age_).long().cuda()
            second_age = Variable(second_age_).long().cuda()

            diff_age_pro = second_age
            diff_age_reg = first_age

            with torch.no_grad():
                style = MappingNetwork(diff_age_pro)
                in_f_val, c4_val, c3_val, c2_val, c1_val = Encoder(first_img)
                in_f_val_, c4_val, c3_val, c2_val, c1_val, _, _ = IEM(in_f_val, c4_val, c3_val, c2_val, c1_val)
                mask_second = Generator(in_f_val_, c4_val, c3_val, c2_val, c1_val, style)
                gen_second_img = mask_second + first_img

                out_first_age = Age_predictor(in_f_val)



                """
                MAE
                """
                # calculate 1st MAE
                y_test1, bc_test1 = num2vect(first_age_, bin_range, bin_step, sigma)
                x_test1 = out_first_age[0].reshape([batch_size, -1])
                x_test1 = x_test1.cpu().numpy()
                prob_test1 = np.exp(x_test1)
                pred_test_01 = prob_test1 @ bc_test1
                pred_test1 = torch.tensor(pred_test_01)
                mae_1st = torch.abs(pred_test1.cuda() - first_age).sum() / batch_size
                MAE_1st += mae_1st



        """
        Early Stopping
        """
        # Save the best model in memory

        if not_improve < 20:

            MAE_total = MAE_1st
            MAE_validation = MAE_total / (120 / batch_size)

            if min_mae > MAE_validation:
                print('MAE Decreasing.. {:.3f} >> {:.3f} '.format(min_mae, MAE_validation))
                min_mae = MAE_validation
                not_improve = 0

                # Save the weights of the encoder and age_predictor in memory
                best_models['Encoder'] = Encoder.state_dict()
                best_models['Age_predictor'] = Age_predictor.state_dict()

            else:
                not_improve += 1
                print(f'MAE Not Decrease for {not_improve} time')
                if not_improve == 20:
                    print('MAE not decrease for 20 times, Stop Age Prediction Training')

        else:
            pass



        print('D_loss:{:.3f} G_loss:{:.3f} A_loss:{:.3f} N_loss:{:.3f} I_loss:{:.3f}'
              .format(D_loss / (train_loop / batch_size),
                      G_loss / (train_loop / batch_size),
                      A_loss / (train_loop / batch_size),
                      N_loss / (train_loop / batch_size),
                      I_loss / (train_loop / batch_size)))

        print('MAE_real:{:3f} MAE_inter:{:.3f} MAE_final:{:.3f}'
              .format(MAE_real / (train_loop / batch_size),
                      MAE_inter / (train_loop / batch_size),
                      MAE_final / (train_loop / batch_size)))


        Encoder.train()
        IEM.train()
        Generator.train()
        Age_predictor.train()
        MappingNetwork.train()


        # # Path for Server
        save_dir = '../save_dir/'


        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if epoch > 99 and epoch % 10 == 0:
            torch.save(best_models['Encoder'], save_dir + '/Encoder_epoch_%d.pth' % epoch)
            torch.save(IEM.state_dict(), save_dir + '/IEM_epoch_%d.pth' % epoch)
            torch.save(Generator.state_dict(), save_dir + '/Decoder_epoch_%d.pth' % epoch)
            torch.save(best_models['Age_predictor'], save_dir + '/Age_predictor_%d.pth' % epoch)
            torch.save(MappingNetwork.state_dict(), save_dir + '/MappingNetwork_%d.pth' % epoch)


if __name__ == '__main__':
    main()