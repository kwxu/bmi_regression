import datetime
import torch
import os.path as osp
import os
from tools.utils import mkdir_p
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, auc
import pytz
import numpy as np
from model.grad_cam import GradCAM
import nibabel as nib
from skimage.transform import resize
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from tools.utils import get_logger
from tools.data_io import ScanWrapper


logger = get_logger('Trainer')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


def get_validation_statics(label, predicted_prob):
    fpr, tpr, _ = roc_curve(label, predicted_prob, pos_label=1)
    precision, recall, _ = precision_recall_curve(label, predicted_prob, pos_label=1)
    roc_auc = roc_auc_score(label, predicted_prob)
    prc_auc = auc(recall, precision)

    summary_item = {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
        'label': label,
        'pred': predicted_prob
    }

    return summary_item


def output_predict_array(performance_statics, out_csv):
    data_dict = {
        'file_name': performance_statics['file_name_list'],
        'pred': performance_statics['pred_list'][:, 0],
        'target': performance_statics['target_list'][:, 0]
    }

    print(data_dict)
    data_df = pd.DataFrame(data_dict)
    print(f'Save prediction to {out_csv}')
    data_df.to_csv(out_csv, index=False)


def get_mean_validation_statics_statical_approach(valid_result_array):
    num_fold = len(valid_result_array)
    accuracy_array = np.zeros((num_fold,), dtype=float)
    roc_auc_array = np.zeros((num_fold,), dtype=float)

    for idx_fold in range(num_fold):
        summary_statics = valid_result_array[idx_fold]
        pred = summary_statics['pred'].astype(float)
        label = summary_statics['label'].astype(int)
        # print(label.shape)
        # print(pred.shape)
        accuracy_array[idx_fold] = accuracy_score(label, pred.round())
        roc_auc_array[idx_fold] = summary_statics['roc_auc']

    accuracy_mean = np.mean(accuracy_array)
    accuracy_std = np.std(accuracy_array)
    auc_mean = np.mean(roc_auc_array)
    auc_std = np.std(roc_auc_array)

    print(f'CV mean accuracy: {accuracy_mean}, {accuracy_std}')
    print(f'CV mean auc: {auc_mean}, {auc_std}')

def get_mean_validation_statics_for_cv_array(valid_result_array):
    label = []
    pred = []

    for valid_result in valid_result_array:
        label.append(valid_result['label'])
        pred.append(valid_result['pred'])

    label = np.concatenate(label)
    pred = np.concatenate(pred)

    fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
    precision, recall, _ = precision_recall_curve(label, pred, pos_label=1)
    roc_auc = roc_auc_score(label, pred)
    prc_auc = auc(recall, precision)

    summary_item = {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'prc_auc': prc_auc,
        'label': label,
        'pred': pred
    }

    return summary_item


class Trainer(object):
    def __init__(
            self,
            cuda,
            model,
            optimizer=None,
            train_loader=None,
            validate_loader=None,
            test_loader=None,
            train_root_dir=None,
            out=None,
            max_epoch=None,
            batch_size=None,
            config=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.test_loader = test_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

        self.train_root_dir = train_root_dir
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.max_epoch = max_epoch
        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0
        self.batch_size = batch_size
        self.config = config

        self.optimal_model_path = osp.join(self.out, 'model_optimal.pth')
        self.validate_performance = None
        self.test_performance = None

    def train(self):
        self.model.train()
        # self.model.float()
        out = osp.join(self.out, 'training')
        mkdir_p(out)

        pred_history = []
        target_history = []
        epoch_averaged_loss_history = []
        sofar = 0
        for batch_idx, (data, target, file_name) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.optim.zero_grad()
            pred_val = self.model(data)
            loss = nn.MSELoss()(pred_val, target.float())

            # print(pred_val.data.cpu().numpy())
            # print(target.data.cpu().numpy())

            if ((batch_idx > 0) & (batch_idx % 10 == 0)):
                averaged_loss = np.sum(np.array(epoch_averaged_loss_history)) / sofar
                logger.info(f'epoch={self.epoch}, batch_idx={batch_idx}, averaged_loss={averaged_loss:.5f}\n')

            epoch_averaged_loss_history.append(loss.data.item() * data.size(0))
            sofar += data.size(0)
            pred_history += pred_val.data.cpu().numpy().tolist()
            target_history += target.data.cpu().numpy().tolist()

            loss.backward()
            self.optim.step()

        # Averaged loss
        for idx in range(len(pred_history)):
            pred = pred_history[idx]
            target = target_history[idx]
            print(f'[{pred}, {target}]')
        # print(pred_history)
        # print(target_history)
        epoch_averaged_loss = np.sum(np.array(epoch_averaged_loss_history)) / sofar
        logger.info(f'epoch={self.epoch}, averaged_loss={epoch_averaged_loss:.5f}\n')
        log_file4 = osp.join(out, 'train_epoch_loss.txt')
        fv4 = open(log_file4, 'a')
        fv4.write(f'{self.epoch} {epoch_averaged_loss:.5f}\n')
        fv4.close()

    def validate(self, run_test=False):
        self.model.eval()
        data_loader = None
        if run_test:
            data_loader = self.test_loader
            out = osp.join(self.out, 'test')
        else:
            data_loader = self.validate_loader
            out = osp.join(self.out, 'validation')

        mkdir_p(out)

        pred_history = []
        target_history = []
        loss_history = []
        file_name_list = []
        sofar = 0

        perform_statics = None
        # with torch.no_grad():
        for batch_idx, (data,target,file_name) in tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=80,
                leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            pred_val = self.model(data)

            test_loss = nn.MSELoss()(pred_val, target.float())

            sofar += data.size(0)

            loss_history.append(test_loss.data.cpu().numpy().tolist() * data.size(0))
            pred_history += pred_val.data.cpu().numpy().tolist()
            target_history += target.data.cpu().numpy().tolist()
            file_name_list += list(file_name)

        averaged_loss = np.sum(np.array(loss_history)) / sofar

        performance_data = {
            'loss': averaged_loss,
            'pred_list': np.array(pred_history),
            'target_list': np.array(target_history),
            'file_name_list': file_name_list
        }

        if run_test:
            self.test_performance = performance_data.copy()

            fv_str = f'Test loss: {averaged_loss:.5f} \n'
            test_performance_statics = osp.join(out, 'test_perform')
            fv_test_performance = open(test_performance_statics, 'w')
            fv_test_performance.write(fv_str)
            fv_test_performance.close()
            pridict_out_csv = os.path.join(out, 'predict.csv')
            output_predict_array(self.test_performance, pridict_out_csv)
        else:
            self.validate_performance = performance_data.copy()

            log_file4 = osp.join(out, 'validation_epoch_loss.txt')
            fv4 = open(log_file4, 'a')
            fv4.write(f'{self.epoch} {averaged_loss:.5f}\n')
            fv4.close()

    def train_epoch(self):
        p_lr_scheduler = ('lr_scheduler_step' in self.config)
        lr_scheduler = None
        if p_lr_scheduler:
            lr_scheduler = StepLR(self.optim,
                                  step_size=self.config['lr_scheduler_step'],
                                  gamma=self.config['lr_scheduler_gamma'])

        optimal_validate_mse = float("inf")
        optimal_validate_epoch = 0
        no_increase_step = 0
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            out = osp.join(self.out, 'models')
            mkdir_p(out)

            model_pth = '%s/model_epoch_%04d.pth' % (out, epoch)

            if os.path.exists(model_pth):
                if self.cuda:
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                # self.model.load_state_dict(torch.load(model_pth))
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
                # if epoch % 5 == 0:
                self.validate()
                if lr_scheduler:
                    lr_scheduler.step()
            else:
                # self.validate()
                self.train()
                # if epoch % 5 == 0:
                self.validate()
                if self.config['output_epoch_model']:
                    torch.save(self.model.state_dict(), model_pth)
                if lr_scheduler:
                    lr_scheduler.step()

            cur_validate_mse = self.validate_performance['loss']
            if cur_validate_mse < optimal_validate_mse:
                print(f'Updating optimal epoch. Epoch: {epoch}, MSE: {cur_validate_mse:.5f}')
                print(f'Updatting {self.optimal_model_path}')
                torch.save(self.model.state_dict(), self.optimal_model_path)
                no_increase_step = 0
                optimal_validate_mse = cur_validate_mse
                optimal_validate_epoch = epoch
            else:
                print(f'No increase step. Epoch: {epoch}, MSE: {cur_validate_mse:.5f}')
                no_increase_step += 1

            no_increase_step_limit = self.config['no_increase_step_limit']
            if no_increase_step > no_increase_step_limit:
                print(f'No increase for validate in {no_increase_step_limit} steps. Terminate training.')
                print(f'Optimal validate MSE: {optimal_validate_mse} at epoch: {optimal_validate_epoch}')
                break

    def run_test(self):
        # load optimal model
        self.model.load_state_dict(torch.load(self.optimal_model_path, map_location=lambda storage, location: storage))
        self.validate(run_test=True)

    def _run_grad_cam_loader(self, data_loader, out_folder):
        mkdir_p(out_folder)

        for batch_idx, (data,target,sub_name) in tqdm.tqdm(
                enumerate(data_loader), total=len(data_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            gcam = GradCAM(model=self.model)
            gcam.forward(data)
            one_hot_ids = np.zeros((len(data), 1), dtype=int)
            one_hot_ids[:, 0] = 0
            one_hot_ids = torch.from_numpy(one_hot_ids)
            if self.cuda:
                one_hot_ids = one_hot_ids.cuda()
            gcam.backward(ids=one_hot_ids)
            regions = gcam.generate(target_layer=self.config['gcam_target_layer'])
            targets = target.data.cpu().numpy()

            file_names = list(sub_name)
            in_img_file_paths = [os.path.join(self.config['input_img_dir'], file_name) for file_name in file_names]

            for idx_scan in range(len(data)):
                file_name = file_names[idx_scan]
                out_img_path = osp.join(
                    out_folder,
                    file_name
                )
                in_img = ScanWrapper(in_img_file_paths[idx_scan])
                out_heat_map = regions.cpu().numpy()[idx_scan, 0]
                out_heat_map = np.transpose(out_heat_map, (1, 2, 0))
                in_img.save_scan_same_space(out_img_path, out_heat_map)
                del in_img, out_heat_map

            # gcam.backward_del(ids=ids[:, [0]])
            gcam.backward_del(ids=one_hot_ids)
            # del gcam, probs, ids, regions
            del gcam, regions

            torch.cuda.empty_cache()
            # gc.collect()

    def run_grad_cam(self):
        self.model.load_state_dict(torch.load(self.optimal_model_path, map_location=lambda storage, location: storage))
        self.model.eval()

        out = osp.join(self.out, 'grad_CAM')
        mkdir_p(out)

        layer_name = self.config['gcam_target_layer']
        self._run_grad_cam_loader(self.test_loader, osp.join(out, f'test.{layer_name}'))