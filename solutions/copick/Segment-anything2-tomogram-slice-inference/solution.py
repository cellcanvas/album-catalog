###album catalog: cellcanvas

from album.runner.api import get_args, setup

env_file = """
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit
  - pytorch-cuda==12.1
  - pip:
    - matplotlib
    - opencv-python
    - copick==0.5.5
    - git+https://github.com/facebookresearch/segment-anything-2.git
"""

def run():
    import numpy as np
    import torch
    import zarr
    import matplotlib.pyplot as plt
    from copick.impl.filesystem import CopickRootFSSpec
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    import os

    # get the current working directory
    current_working_directory = os.getcwd()
    print(f'current_working_directory {current_working_directory}')

    # CLI arguments
    args = get_args()
    copick_config_path = args.copick_config_path
    run_name = args.run_name
    tomo_type = args.tomo_type
    slice_index = int(args.slice_index)
    voxel_spacing = args.voxel_spacing
    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = args.model_cfg
    logdir = args.logdir


    def normalize_img(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_array = (arr - min_val) / (max_val - min_val) * 255
        return normalized_array.astype(np.uint8)
    
    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 
        ax.imshow(img)


    def save_img(image, name, masks=None):
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        if masks is not None:
            show_anns(masks)
            
        height, width, _ = image.shape
        dpi = 100
        fig = plt.gcf()
        fig.set_size_inches(width / dpi, height / dpi)
        plt.savefig(f'{name}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    root = CopickRootFSSpec.from_file(copick_config_path)
    run = root.get_run(run_name)
    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    image = zarr.open(tomogram.zarr(), mode='r')['0']
    image = image[slice_index,:,:]
    image = normalize_img(image)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    
    # checkpoint_version = sam2_checkpoint.split('/')[-1]
    # if checkpoint_version == "sam2_hiera_tiny.pt":
    #     model_cfg = "sam2_hiera_t.yaml"
    # elif checkpoint_version == "sam2_hiera_small.pt":
    #     model_cfg = "sam2_hiera_s.yaml"
    # elif checkpoint_version == "sam2_hiera_base_plus.pt":
    #     model_cfg = "sam2_hiera_b+.yaml"
    # elif checkpoint_version == "sam2_hiera_large.pt":
    #     model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    masks = mask_generator.generate(image)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    save_img(image, f'{logdir}/segmented_slice_{run_name}', masks=masks)
    save_img(image, f'{logdir}/slice_{run_name}')


setup(
    group="copick",
    name="Segment-anything2-tomogram-slice-inference",
    version="0.0.1",
    title="Generate segmentation masks for Copick Dataset using Segment Anything 2",
    description="Automatically generate segmentation masks for the Copick dataset. Install on a device with GPU access. Need cuda/12.1.1_530.30.02 and cudnn/8.9.7.29_cuda12 to compile.",
    solution_creators=["Zhuowen Zhao, Kyle Harrington"],
    cite=[{"text": "Meta AI team.", "url": "https://ai.meta.com/blog/segment-anything-2/"}],
    tags=["imaging", "segmentation", "cryoet", "Python", "SAM 2"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "description": "Path to the Copick configuration file",
            "type": "string",
            "required": True,
        },
        {
            "name": "run_name",
            "description": "Name of the run in the Copick project for inferencing",
            "type": "string",
            "required": True,
        },
        {
            "name": "tomo_type",
            "description": "Tomogram type in the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "voxel_spacing",
            "description": "Voxel spacing for the Copick project",
            "type": "float",
            "required": True,
        },
        {
            "name": "slice_index",
            "description": "Z-index of the tomogram slice",
            "type": "string",
            "required": True,
        },
        {
            "name": "sam2_checkpoint",
            "description": "Path to the pre-downloaded SAM2 checkpoints",
            "type": "string",
            "required": True,
        },
        {
            "name": "model_cfg",
            "description": "Model configuration file corresponds to the SAM2 checkpoints",
            "type": "string",
            "required": True,
        },
        {
            "name": "logdir",
            "description": "Output directory name in the current working directory. Default is outputs",
            "type": "string",
            "required": False,
            "default": "outputs",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)