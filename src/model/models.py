from model.trainer import print_network
from model.ResNet3D import resnet101, resnet34, resnet10, resnet18


def create_model(config):
    add_intensity_map = config['add_intensity_map']
    add_jac_map = config['add_jacobian_map']
    add_d_index_map = config['add_d_index_map']
    add_valid_mask = config['add_valid_mask_map']
    sample_size = config['sample_size']
    sample_duration = config['sample_duration']
    fcnum = config['fcnum']
    networkName = config['networkName']
    resnet_shortcut = 'B'
    # Regression problem
    clss_num = 1
    model = None
    num_in_channel = 0
    if add_intensity_map:
        num_in_channel += 1
    if add_jac_map:
        num_in_channel += 1
    if add_valid_mask:
        num_in_channel += 1
    if add_d_index_map:
        num_in_channel += 1
    if networkName == 'resnet101':
        model = resnet101(
            num_classes=clss_num,
            shortcut_type=resnet_shortcut,
            sample_size=sample_size,
            sample_duration=sample_duration,
            fcnum=fcnum,
            in_channel=num_in_channel)
    elif networkName == 'resnet34':
        model = resnet34(
            num_classes=clss_num,
            shortcut_type=resnet_shortcut,
            sample_size=sample_size,
            sample_duration=sample_duration,
            fcnum=fcnum,
            in_channel=num_in_channel)
    elif networkName == 'resnet18':
        model = resnet18(
            num_classes=clss_num,
            shortcut_type=resnet_shortcut,
            sample_size=sample_size,
            sample_duration=sample_duration,
            fcnum=fcnum,
            in_channel=num_in_channel)
    elif networkName == 'resnet10':
        model = resnet10(
            num_classes=clss_num,
            shortcut_type=resnet_shortcut,
            sample_size=sample_size,
            sample_duration=sample_duration,
            fcnum=fcnum,
            in_channel=num_in_channel)
    else:
        raise NotImplementedError

    print_network(model)

    return model