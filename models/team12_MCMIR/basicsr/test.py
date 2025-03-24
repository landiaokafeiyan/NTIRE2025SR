import logging
import torch
import os
import math
from os import path as osp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs,tensor2img
from basicsr.utils.options import dict2str, parse_options
from dataloader_moa import *

def tile_process(img, model, scale, tile_size, tile_pad):
    """
    Processes an input image (C, H, W) by dividing it into tiles, upscaling each tile using a model,
    and then merging the processed tiles back into a single output image.

    Parameters:
    - img (torch.Tensor): Input image tensor of shape (C, H, W), values in range [0,1].
    - model (torch.nn.Module): The deep learning model used for upscaling.
    - scale (int): The upscaling factor.
    - tile_size (int): The size of each tile.
    - tile_pad (int): The padding applied to tiles.

    Returns:
    - torch.Tensor: The processed and upscaled output image in (C, H, W), range **[0,255]**.
    """
    _, height, width = img.shape
    output_height = height * scale
    output_width = width * scale

    # Initialize output image (keep float32, convert to 255 at the end)
    output = torch.zeros((3, output_height, output_width), dtype=torch.float32, device=img.device)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # Loop through all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate tile offsets
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            # Define the input tile area within the original image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # Define the padded tile area
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # Extract the input tile with padding
            input_tile = img[:, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # Model expects (B, C, H, W) format
            input_tile = input_tile.unsqueeze(0)  # Add batch dimension

            # Perform upscaling on the tile
            try:
                with torch.no_grad():
                    model.feed_test_data(input_tile)
                    model.test()

                    visuals = model.get_current_visuals()
                    output_tile = visuals['result']  # Output remains (B, C, H, W)
                    
                    # Remove batch dimension
                    output_tile = output_tile.squeeze(0)

                    # Release memory
                    del model.lq
                    del model.output
                    torch.cuda.empty_cache()

            except RuntimeError as error:
                print(f'Error processing tile {y * tiles_x + x + 1}/{tiles_x * tiles_y}: {error}')
                continue  # Skip this tile if an error occurs
            
            print(f'Processing Tile {y * tiles_x + x + 1}/{tiles_x * tiles_y}')

            # Define the output tile position in the final image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Define the valid (unpadded) portion of the output tile
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * scale

            # Place the processed tile into the output image
            output[:, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    # Convert from [0,1] to [0,255] before returning
    output = (output * 255).clamp(0, 255).byte()  # Ensure valid range and convert to uint8

    return output  # Returns tensor in (C, H, W), range [0,255]

def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    # test_loaders = []
    # for _, dataset_opt in sorted(opt['datasets'].items()):
    #     test_set = build_dataset(dataset_opt)
    #     test_loader = build_dataloader(
    #         test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
    #     logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
    #     test_loaders.append(test_loader)
    
    # DIV2K dataset and loader:
    test_set = TestSetLoader(args)
    dataloader = DataLoader(
                dataset=test_set,
                batch_size=opt['datasets']['train']['batch_size_per_gpu'] * torch.cuda.device_count(),
                shuffle=False,
                num_workers=4,
            )
    # create model
    model = build_model(opt)
    # if use_pbar:
    #     pbar = tqdm(total=len(dataloader), unit='image')
    output_dir = args.data_dir + "/HR/"
    
    for imgs, paths in dataloader:
        for test_img, img_path in zip(imgs, paths):     
            sr_image = tile_process(test_img, model, opt['scale'], args.img_size, 0)
            
            if len(sr_image.shape) == 4:  # (B, C, H, W) case
                sr_image = sr_image[0]
                
            sr_image = sr_image.cpu().detach().numpy()
            sr_image = np.transpose(sr_image, (1, 2, 0))
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
            img_name = os.path.basename(img_path)
            img_name = img_name[0:4]+img_name[6:]
            # Save the transformed image
            output_path = output_dir+img_name
            cv2.imwrite(output_path, sr_image)
            del sr_image
            print(f"Image saved at {output_path}")
                
    # for test_loader in test_loaders:
    #     test_set_name = test_loader.dataset.opt['name']
    #     logger.info(f'Testing {test_set_name}...')
    #     model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

def main(model_dir, input_path, output_path, device=None):
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    opt['path']['pretrain_network_g'] = model_dir
    opt['datasets']['train']['dataroot_gt'] = os.path.join(input_path,'HR')
    opt['datasets']['train']['dataroot_lq'] = os.path.join(input_path,'LR')
    setattr(args, 'data_dir', input_path)  # Set data directory
    output_dir = output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))
    
    # DIV2K dataset and loader:
    test_set = TestSetLoader(args)
    dataloader = DataLoader(
                dataset=test_set,
                batch_size=opt['datasets']['train']['batch_size_per_gpu'] * torch.cuda.device_count(),
                shuffle=False,
                num_workers=4,
            )
    # create model
    model = build_model(opt)
    # if use_pbar:
    #     pbar = tqdm(total=len(dataloader), unit='image')
    
    for imgs, paths in dataloader:
        for test_img, img_path in zip(imgs, paths):     
            sr_image = tile_process(test_img, model, opt['scale'], args.img_size, 0)
            
            if len(sr_image.shape) == 4:  # (B, C, H, W) case
                sr_image = sr_image[0]
                
            sr_image = sr_image.cpu().detach().numpy()
            sr_image = np.transpose(sr_image, (1, 2, 0))
            sr_image = cv2.cvtColor(sr_image, cv2.COLOR_BGR2RGB)
            img_name = os.path.basename(img_path)
            img_name = img_name[0:4]+img_name[6:]
            # Save the transformed image
            img_path = output_dir+img_name
            cv2.imwrite(img_path, sr_image)
            del sr_image
            print(f"Image saved at {img_path}")
