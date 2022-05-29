import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
import optuna

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class NCLG():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 2:
            model_name = 'inpaint'


        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)


        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        #train mode
        if self.config.MODE == 1:
            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_INPAINT_LANDMARK_FLIST,
                                             config.TRAIN_MASK_FLIST, augment=True, training=True)
                self.val_dataset = Dataset(config, config.VAL_INPAINT_IMAGE_FLIST, config.VAL_INPAINT_LANDMARK_FLIST,
                                           config.TEST_MASK_FLIST, augment=True, training=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
        #optuna mode
        elif self.config.MODE == 5:

            if self.config.MODEL == 2:

                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_INPAINT_LANDMARK_FLIST,
                                             config.TRAIN_MASK_FLIST, augment=True, training=True)
                self.val_dataset = Dataset(config, config.VAL_INPAINT_IMAGE_FLIST, config.VAL_INPAINT_LANDMARK_FLIST,
                                           config.TEST_MASK_FLIST, augment=True, training=True)
                self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)


        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_INPAINT_LANDMARK_FLIST, config.TEST_MASK_FLIST,
                                            augment=False, training=False)


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 2:
            self.inpaint_model.load()


    def save(self):
        if self.config.MODEL == 2:
            self.inpaint_model.save()


    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )


        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])


            for items in train_loader:

                self.inpaint_model.train()

                if model == 2:
                    images, landmarks, masks = self.cuda(*items)

                landmarks[landmarks >= self.config.INPUT_SIZE] = self.config.INPUT_SIZE - 1
                landmarks[landmarks < 0] = 0
                # inpaint model

                if model == 2:
                    landmarks[landmarks>=self.config.INPUT_SIZE] = self.config.INPUT_SIZE-1
                    landmarks[landmarks<0] = 0

                    outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss = self.inpaint_model.process(images,landmarks,masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                ## visialization
                if iteration % 10 == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )

                    path_masked = os.path.join(self.results_path,self.model_name,'masked')
                    path_result = os.path.join(self.results_path, self.model_name,'result')
                    path_joint = os.path.join(self.results_path,self.model_name,'joint')
                    path_landmark_mask = os.path.join(self.results_path, self.model_name, 'landmark_mask')
                    path_landmark_gt = os.path.join(self.results_path, self.model_name, 'landmark_gt')
                    name = self.train_dataset.load_name(epoch-1)[:-4]+'.png'

                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)
                    create_dir(path_landmark_mask)
                    create_dir(path_landmark_gt)
                    landmark_mask_image = images * (1 - masks) + masks

                    landmark_mask_image = (landmark_mask_image[0].squeeze().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')
                    landmark_mask_image = landmark_mask_image.copy()
                    landmark_gt = landmark_mask_image.copy()

                    for i in range(outputs_lmk.shape[1]):
                        circle(landmark_mask_image, (int(outputs_lmk[0, i, 0]), int(outputs_lmk[0, i, 1])), radius=2,
                               color=(0, 255, 0), thickness=-1)
                    for i in range(landmarks.shape[1]):
                        circle(landmark_gt, (int(landmarks[0, i, 0]), int(landmarks[0, i, 1])), radius=2,
                               color=(0, 255, 0), thickness=-1)

                    masked_images = self.postprocess(images*(1-masks)+masks)[0]
                    images_result = self.postprocess(outputs_merged)[0]

                    print(os.path.join(path_joint,name[:-4]+'.png'))
                    landmark_mask_image = Image.fromarray(landmark_mask_image)
                    landmark_mask_image.save(os.path.join(path_landmark_mask, name))

                    landmark_gt = Image.fromarray(landmark_gt)
                    landmark_gt.save(os.path.join(path_landmark_gt, name))

                    images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                    imsave(masked_images,os.path.join(path_masked,name))
                    imsave(images_result,os.path.join(path_result,name))

                    print(name + ' complete!')
                ########


                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)
                # sample model at checkpoints

                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    # print('\nstart sample...\n')
                    outputs_img_val, outputs_lmk_val, gen_loss_val, dis_loss_val, logs_val, gen_gan_loss_val, gen_l1_loss_val, gen_content_loss_val, gen_style_loss_val, tv_loss_val, lmk_loss_val = self.sample()
                    print('g_Loss_val = %f' % (gen_loss_val.item()))
                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0 and self.config.MODEL == 2:
                    print('\nstart eval...\n')
                    self.eval()
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')

    def test(self):

        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        print('here')
        index = 0
        for items in test_loader:
            images, landmarks, masks = self.cuda(*items)
            index += 1

            if model == 2:
                landmarks[landmarks >= self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
                landmarks[landmarks < 0] = 0

                inputs = (images * (1 - masks))

                outputs_img, outputs_lmk = self.inpaint_model(images, masks)
                outputs_lmk *= self.config.INPUT_SIZE
                outputs_lmk = outputs_lmk.reshape((-1, self.config.LANDMARK_POINTS, 2))

                outputs_merged = (outputs_img * masks) + (images * (1 - masks))

                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_masked = os.path.join(self.results_path,self.model_name,'masked')
                path_result = os.path.join(self.results_path, self.model_name,'result')
                path_joint = os.path.join(self.results_path,self.model_name,'joint')
                path_landmark_mask = os.path.join(self.results_path, self.model_name, 'landmark_mask')
                path_landmark_output = os.path.join(self.results_path, self.model_name, 'landmark_output')
                path_landmark_only = os.path.join(self.results_path, self.model_name, 'landmark_only')

                name = self.test_dataset.load_name(index-1)[:-4]+'.png'

                create_dir(path_masked)
                create_dir(path_result)
                create_dir(path_joint)
                create_dir(path_landmark_mask)
                create_dir(path_landmark_output)
                create_dir(path_landmark_only)
                landmark_mask_image = images * (1 - masks) + masks
                landmark_mask_image = (landmark_mask_image.squeeze().cpu().numpy().transpose(1,2,0)*255).astype('uint8')
                landmark_mask_image = landmark_mask_image.copy()
                for i in range(landmarks.shape[1]):
                    circle(landmark_mask_image, (int(landmarks[0, i, 0]), int(landmarks[0, i, 1])), radius=2,
                           color=(0, 255, 0), thickness=-1)

                landmark_output_image = outputs_img
                landmark_output_image = (landmark_output_image.squeeze().cpu().detach().numpy().transpose(1,2,0)*255).astype('uint8')
                landmark_output_image = landmark_output_image.copy()
                for i in range(outputs_lmk.shape[1]):
                    circle(landmark_output_image, (int(outputs_lmk[0, i, 0]), int(outputs_lmk[0, i, 1])), radius=2,
                           color=(0, 255, 0), thickness=-1)

                landmark_map = torch.zeros(1,3,self.config.INPUT_SIZE, self.config.INPUT_SIZE)
                landmark_map = (landmark_map.squeeze().cpu().detach().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                landmark_map = np.array(landmark_map)
                landmark_map = landmark_map.copy()

                for i in range(outputs_lmk.shape[1]):
                    circle(landmark_map,(int(outputs_lmk[0, i, 0]), int(outputs_lmk[0, i, 1])), radius=2, color=(0,255, 0), thickness=-1)

                masked_images = self.postprocess(images*(1-masks)+masks)[0]
                images_result = self.postprocess(outputs_merged)[0]

                print(os.path.join(path_joint,name[:-4]+'.png'))
                landmark_mask_image = Image.fromarray(landmark_mask_image)
                landmark_mask_image.save(os.path.join(path_landmark_mask, name))

                landmark_output_image = Image.fromarray(landmark_output_image)
                landmark_output_image.save(os.path.join(path_landmark_output, name))

                landmark_map = Image.fromarray(landmark_map)
                landmark_map.save(os.path.join(path_landmark_only,name))

                images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                imsave(masked_images,os.path.join(path_masked,name))
                imsave(images_result,os.path.join(path_result,name))

                print(name + ' complete!')

        print('\nEnd Testing')



    def sample(self, it=None):
        self.inpaint_model.eval()

        model = self.config.MODEL

        items = next(self.sample_iterator)

        if model == 2:
            images,landmarks,masks = self.cuda(*items)

        landmarks[landmarks>=self.config.INPUT_SIZE-1] = self.config.INPUT_SIZE-1
        landmarks[landmarks<0] = 0


        # inpaint model
        if model == 2:


            iteration = self.inpaint_model.iteration
            inputs = (images * (1 - masks)) + masks
            for i in range(inputs.shape[0]):
                inputs[i, :, landmarks[i, 0:self.config.LANDMARK_POINTS, 1], landmarks[i, 0:self.config.LANDMARK_POINTS, 0]] = 1-masks[i,0,landmarks[i, :, 1], landmarks[i,:,0]]

            outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss = self.inpaint_model.process(
                images, landmarks, masks)
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))


        if it is not None:
            iteration = it


        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1


        elif model == 2:
            images = stitch_images(
                self.postprocess(images),
                self.postprocess(inputs),
                self.postprocess(outputs_img),
                self.postprocess(outputs_merged),
                img_per_row=image_per_row
            )



        if iteration % 200 == 0:
            path = os.path.join(self.samples_path, self.model_name)
            name = os.path.join(path, str(iteration).zfill(5) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            images.save(name)

        return outputs_img, outputs_lmk, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss, tv_loss, lmk_loss

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):

        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

