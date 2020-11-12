import argparse
from tools.utils import read_file_contents_list
from tools.clinical import ClinicalDataReaderSPORE
from tools.utils import mkdir_p
import torch
from model.trainer import Trainer
from model.dataloader import get_data_loader_cv
from model.models import create_model
import yaml
import os
import numpy as np
from tools.plot import plot_cv_roc
from tools.utils import get_logger


logger = get_logger('train_n_fold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', type=str, default='simg_bmi_regression_10_nfs.yaml')
    parser.add_argument('--run-train', type=str, default='True')
    parser.add_argument('--run-test', type=str, default='False')
    parser.add_argument('--run-grad-cam', type=str, default='False')
    parser.add_argument('--train-fold', type=int, default=0)
    args = parser.parse_args()

    SRC_ROOT = os.path.dirname(os.path.realpath(__file__)) + '/..'
    yaml_config = os.path.join(SRC_ROOT, f'src/yaml/{args.yaml_config}')
    logger.info(f'Read yaml file {yaml_config}')
    f = open(yaml_config, 'r').read()
    config = yaml.safe_load(f)

    out_folder = config['exp_dir']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epoch_num = config['epoch_num']
    fold_num = config['fold_num']

    mkdir_p(out_folder)

    # load CUDA
    cuda = torch.cuda.is_available()
    print(f'cuda: {cuda}')
    # cuda = False
    torch.manual_seed(1)

    # Create data loader
    train_loader_list, valid_loader_list, test_loader_list = get_data_loader_cv(config)

    # Create trainer list
    performance_array = []
    for idx_fold in range(fold_num):
        # If train only one fold
        if args.train_fold != -1:
            # Only train on specified fold.
            if args.train_fold != idx_fold:
                continue

        # Create model
        model = create_model(config)
        if cuda:
            torch.cuda.manual_seed(1)
            model = model.cuda()

        # load optimizor
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        # Create trainer
        fold_out_folder = os.path.join(out_folder, f'fold_{idx_fold}')
        train_loader = train_loader_list[idx_fold]
        validate_loader = valid_loader_list[idx_fold]
        test_loader = test_loader_list[idx_fold]
        trainer_obj = Trainer(
            cuda,
            model,
            optimizer=optim,
            train_loader=train_loader,
            validate_loader=validate_loader,
            test_loader=test_loader,
            out=fold_out_folder,
            max_epoch=epoch_num,
            batch_size=batch_size,
            config=config
        )

        # Train
        trainer_obj.epoch = config['start_epoch']
        if args.run_train == 'True':
            trainer_obj.train_epoch()

        # Test
        if args.run_test == 'True':
            trainer_obj.run_test()
            performance_array.append(trainer_obj.test_performance)

        if args.run_grad_cam == 'True':
            trainer_obj.run_grad_cam()

    if args.run_test == 'True':
        mse_array = np.array([statics_dict['loss'] for statics_dict in performance_array])
        rmse_array = np.sqrt(mse_array)
        rmse_mean = np.mean(rmse_array)
        rmse_std = np.std(rmse_array)
        perf_str = f'RMSE {rmse_mean:.5f} ({rmse_std:.5f})\n'
        print(f'Performance of cross-validation:')
        print(perf_str)
        perf_file = os.path.join(out_folder, 'perf')
        with open(perf_file, 'w') as fv:
            fv.write(perf_str)
            fv.close()


if __name__ == '__main__':
    main()