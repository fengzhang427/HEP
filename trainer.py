from models.NDM_model import MsImageDis, Dis_content, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg19, get_scheduler
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import torch.nn as nn
from models.GaussianSmoothLayer import GaussionSmoothLayer, GradientLoss
from models.loss import Exposure_control_loss, Color_constancy_loss
import os


##################################################################################
# UNIT
##################################################################################
class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_x = VAEGen(hyperparameters['input_dim_x'], hyperparameters['gen'])  # auto-encoder for domain x
        self.gen_y = VAEGen(hyperparameters['input_dim_y'], hyperparameters['gen'])  # auto-encoder for domain y
        self.dis_x = MsImageDis(hyperparameters['input_dim_x'], hyperparameters['dis'])  # discriminator for domain x
        self.dis_y = MsImageDis(hyperparameters['input_dim_y'], hyperparameters['dis'])  # discriminator for domain y
        self.dis_content = Dis_content()
        self.gpu_id = hyperparameters['gpuID']
        # add background discriminator for each domain
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_x.parameters()) + list(self.dis_y.parameters())
        gen_params = list(self.gen_x.parameters()) + list(self.gen_y.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.content_opt = torch.optim.Adam(self.dis_content.parameters(), lr=lr / 2., betas=(beta1, beta2),
                                            weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters)

        # Network weight initialization
        self.gen_x.apply(weights_init(hyperparameters['init']))
        self.gen_y.apply(weights_init(hyperparameters['init']))
        self.dis_x.apply(weights_init('gaussian'))
        self.dis_y.apply(weights_init('gaussian'))
        self.dis_content.apply(weights_init('gaussian'))

        # initialize the blur network
        self.BGBlur_kernel = [5, 9, 15]
        self.BlurNet = [GaussionSmoothLayer(3, k_size, 25).cuda(self.gpu_id) for k_size in self.BGBlur_kernel]
        self.BlurWeight = [0.25, 0.5, 1]
        self.Gradient = GradientLoss(3, 3)

        # # Load VGG model if needed for test
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg19(13)
            if torch.cuda.is_available():
                self.vgg.cuda(self.gpu_id)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, image, target):
        return torch.mean(torch.abs(image - target))

    def forward(self, x, y):
        self.eval()
        cont_x = self.gen_x.encode_cont(x)
        noise_x = self.gen_x.encode_noise(x)
        cont_y = self.gen_y.encode_cont(y)

        image_x2y = self.gen_y.decode_cont(cont_x)
        h_cat = torch.cat((cont_y, noise_x), 1)
        image_y2x = self.gen_x.decode_recs(h_cat)
        self.train()
        return image_x2y, image_y2x

    def __compute_kl(self, mu):
        # def compute_kl(self, mu, sd):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def content_update(self, x, y, hyperparameters):  #
        # encode
        self.content_opt.zero_grad()
        enc_x = self.gen_x.encode_cont(x)
        enc_y = self.gen_y.encode_cont(y)
        pred_fake = self.dis_content.forward(enc_x)
        pred_real = self.dis_content.forward(enc_y)
        if hyperparameters['gan_type'] == 'lsgan':
            loss_D = torch.mean((pred_fake - 0) ** 2) + torch.mean((pred_real - 1) ** 2)
        elif hyperparameters['gan_type'] == 'nsgan':
            all0 = Variable(torch.zeros_like(pred_fake.data).cuda(self.gpu_id), requires_grad=False)
            all1 = Variable(torch.ones_like(pred_real.data).cuda(self.gpu_id), requires_grad=False)
            loss_D = torch.mean(F.binary_cross_entropy(F.sigmoid(pred_fake), all0) +
                                F.binary_cross_entropy(F.sigmoid(pred_real), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(hyperparameters['gan_type'])
        loss_D.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.dis_content.parameters(), 5)
        self.content_opt.step()

    def gen_update(self, x, y, hyperparameters):
        self.gen_opt.zero_grad()
        self.content_opt.zero_grad()

        # encode
        cont_x = self.gen_x.encode_cont(x)
        cont_y = self.gen_y.encode_cont(y)
        noise_x = self.gen_x.encode_noise(y)

        # decode (within domain)
        h_x_cont = torch.cat((cont_x, noise_x), 1)
        rand_x = torch.randn(h_x_cont.size()).cuda()
        x_recon = self.gen_x.decode_recs(h_x_cont + rand_x)
        rand_y = torch.randn(cont_y.size()).cuda()
        y_recon = self.gen_y.decode_cont(cont_y + rand_y)

        # decode (cross domain)
        h_y2x_cont = torch.cat((cont_y, noise_x), 1)
        image_y2x = self.gen_x.decode_recs(h_y2x_cont + rand_x)
        image_x2y = self.gen_y.decode_cont(cont_x + rand_y)

        # encode again
        cont_y_recon = self.gen_x.encode_cont(image_y2x)
        noise_x_recon = self.gen_x.encode_noise(image_y2x)
        cont_x_recon = self.gen_y.encode_cont(image_x2y)

        # decode again (if needed)
        h_x_cat_recs = torch.cat((cont_x_recon, noise_x_recon), 1)
        image_x2y2x = self.gen_x.decode_recs(h_x_cat_recs) if hyperparameters['recon_cyc_w'] > 0 else None
        image_y2x2y = self.gen_y.decode_cont(cont_y_recon) if hyperparameters['recon_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x = self.recon_criterion(x_recon, x)
        self.loss_gen_recon_y = self.recon_criterion(y_recon, y)
        self.loss_gen_recon_kl_noise = self.__compute_kl(noise_x)
        self.loss_gen_cyc_x = self.recon_criterion(image_x2y2x, x) if image_x2y2x is not None else 0
        self.loss_gen_cyc_y = self.recon_criterion(image_y2x2y, y) if image_y2x2y is not None else 0
        self.loss_gen_recon_kl_cyc_noise = self.__compute_kl(noise_x_recon)
        # GAN loss
        self.loss_gen_adv_x = self.dis_x.calc_gen_loss(image_y2x)
        self.loss_gen_adv_y = self.dis_y.calc_gen_loss(image_x2y)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_x = self.compute_vgg_loss(self.vgg, image_y2x, y) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_y = self.compute_vgg_loss(self.vgg, image_x2y, x) if hyperparameters['vgg_w'] > 0 else 0

        # add background guide loss
        self.loss_bgm = 0
        if hyperparameters['BGM'] != 0:
            for index, weight in enumerate(self.BlurWeight):
                out_y = self.BlurNet[index](image_y2x)
                out_real_y = self.BlurNet[index](y)
                out_x = self.BlurNet[index](image_x2y)
                out_real_x = self.BlurNet[index](x)
                grad_loss_y = self.recon_criterion(out_y, out_real_y)
                grad_loss_x = self.recon_criterion(out_x, out_real_x)
                self.loss_bgm += weight * (grad_loss_x + grad_loss_y)

        # add color constancy loss
        L_color = Color_constancy_loss()
        self.loss_color_x = torch.mean(L_color(image_x2y))
        self.loss_color_y = torch.mean(L_color(image_y2x))

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_x + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_y + \
                              hyperparameters['recon_w'] * self.loss_gen_recon_x + \
                              hyperparameters['recon_w'] * self.loss_gen_recon_y + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_noise + \
                              hyperparameters['recon_cyc_w'] * self.loss_gen_cyc_x + \
                              hyperparameters['recon_cyc_w'] * self.loss_gen_cyc_y + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_x + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_y + \
                              hyperparameters['BGM'] * self.loss_bgm + \
                              hyperparameters['color'] * self.loss_color_x + \
                              hyperparameters['color'] * self.loss_color_y

        self.loss_gen_total.backward()
        self.gen_opt.step()
        self.content_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x, y):
        self.eval()
        x_recon, y_recon, image_y2x, image_x2y = [], [], [], []
        for i in range(x.size(0)):
            cont_x = self.gen_x.encode_cont(x[i].unsqueeze(0))
            cont_y = self.gen_y.encode_cont(y[i].unsqueeze(0))
            noise_x = self.gen_x.encode_noise(x[i].unsqueeze(0))

            h_y2x_cont = torch.cat((cont_y, noise_x), 1)
            h_x2x_cont = torch.cat((cont_x, noise_x), 1)

            x_recon.append(self.gen_x.decode_recs(h_x2x_cont))
            y_recon.append(self.gen_y.decode_cont(cont_y))

            image_y2x.append(self.gen_x.decode_recs(h_y2x_cont))
            image_x2y.append(self.gen_y.decode_cont(cont_x))

        x_recon, y_recon = torch.cat(x_recon), torch.cat(y_recon)
        image_y2x = torch.cat(image_y2x)
        image_x2y = torch.cat(image_x2y)
        self.train()
        return x, x_recon, image_x2y, y, y_recon, image_y2x

    def dis_update(self, x, y, hyperparameters):
        self.dis_opt.zero_grad()
        self.content_opt.zero_grad()

        # encode
        cont_x = self.gen_x.encode_cont(x)
        cont_y = self.gen_y.encode_cont(y)
        noise_x = self.gen_x.encode_noise(x)

        # decode (cross domain)
        h_cat = torch.cat((cont_y, noise_x), 1)
        rand_x = torch.randn(h_cat.size()).cuda()
        image_y2x = self.gen_x.decode_recs(h_cat + rand_x)
        rand_y = torch.randn(cont_x.size()).cuda()
        image_x2y = self.gen_y.decode_cont(cont_x + rand_y)
        # D loss
        self.loss_dis_x = self.dis_x.calc_dis_loss(image_y2x.detach(), x)
        self.loss_dis_y = self.dis_y.calc_dis_loss(image_x2y.detach(), y)
        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_x + self.loss_dis_y)

        self.loss_dis_total.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.dis_content.parameters(), 5)  # dis_content update
        self.dis_opt.step()
        self.content_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.content_scheduler is not None:
            self.content_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_x.load_state_dict(state_dict['x'])
        self.gen_y.load_state_dict(state_dict['y'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_x.load_state_dict(state_dict['x'])
        self.dis_y.load_state_dict(state_dict['y'])

        # load discontent discriminator
        last_model_name = get_model_list(checkpoint_dir, "dis_Content")
        state_dict = torch.load(last_model_name)
        self.dis_content.load_state_dict(state_dict['dis_c'])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.content_opt.load_state_dict(state_dict['dis_content'])

        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        self.content_scheduler = get_scheduler(self.content_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        dis_cont_name = os.path.join(snapshot_dir, 'dis_Content_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'denoise_optimizer.pt')
        torch.save({'x': self.gen_x.state_dict(), 'y': self.gen_y.state_dict()}, gen_name)
        torch.save({'x': self.dis_x.state_dict(), 'y': self.dis_y.state_dict()}, dis_name)
        torch.save({'dis_c': self.dis_content.state_dict()}, dis_cont_name)
        #  opt state
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(),
                    'dis_content': self.content_opt.state_dict()}, opt_name)