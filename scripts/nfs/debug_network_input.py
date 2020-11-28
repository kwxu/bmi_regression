import os
import os.path as osp
from src.tools.utils import get_logger
from tools.utils import mkdir_p
from src.tools.plot import ClipPlotSeriesWithBack
from src.tools.utils import read_file_contents_list, save_file_contents_list, get_logger
from src.tools.data_io import ScanWrapper
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tools.plot import mean_diff_plot, scatter_plot


in_folder = '/nfs/masi/xuk9/SPORE/CAC_class/debug'
out_png_folder = '/nfs/masi/xuk9/SPORE/CAC_class/debug_png'
mkdir_p(out_png_folder)

file_list_txt = '/nfs/masi/xuk9/SPORE/CAC_class/debug/file_list'


def axial_clip_plot_native():
    file_name_list = read_file_contents_list(file_list_txt)
    for file_name in file_name_list:
        in_img_path = os.path.join(in_folder, file_name)
        cliper_obj = ClipPlotSeriesWithBack(
            in_img_path,
            None,
            None,
            10, 35, 15,
            1,
            -255, 255,
            None, None,
            None
        )
        cliper_obj.clip_plot_img_only(out_png_folder)


def main():
    axial_clip_plot_native()


if __name__ == '__main__':
    main()
