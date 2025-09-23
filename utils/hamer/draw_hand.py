import torch
import cv2
import numpy as np
from PIL import Image

from .hamer.models import load_hamer
from .hamer.utils import recursive_to
from .hamer.datasets.vitdet_dataset import ViTDetDataset
from .hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
HAMER_CHECKPOINT = "./utils/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"

class Hamer(object):
    """
    draw 3d hand overlaying pose image
    """
    def __init__(self, cuda_id=0):
        self.device = torch.device('cuda:{}'.format(cuda_id)) if torch.cuda.is_available() else torch.device('cpu')
        model, self.model_cfg = load_hamer(HAMER_CHECKPOINT, 'cuda:{}'.format(cuda_id))

        # Setup HaMeR model
        self.model = model.to(self.device)
        self.model.eval()

        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    def _pil_to_cv2(self, img_pil):
        np_img = np.array(img_pil)
        if img_pil.mode == "RGB":
            img_cv2 = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            img_cv2 = np_img
        return img_cv2

    def _overlay_images(self, base_image, overlay_image):
        # Convert images to RGB and then to numpy arrays
        base_image = base_image.convert("RGB")
        overlay_image = overlay_image.convert("RGB")

        base_array = np.array(base_image)
        overlay_array = np.array(overlay_image)

        # Create a mask where overlay_image is not black ([0, 0, 0])
        mask = np.all(overlay_array != [0, 0, 0], axis=-1)

        # Use the mask to replace the corresponding pixels in base_array
        base_array[mask] = overlay_array[mask]

        return Image.fromarray(base_array)

    def draw_hand(self, pose_img, pose, raw_img):
        W, H = raw_img.size
        pose_img_cv2 = self._pil_to_cv2(pose_img)
        raw_img_cv2 = self._pil_to_cv2(raw_img)

        if isinstance(pose["hands"], dict):
            hand_poses = pose["hands"]["candidate"]
        else:
            hand_poses = pose["hands"]

        bboxes = []
        is_right = []

        # hand_poses: shape (num_hands, num_keypoints, coords...)
        for i, keyp in enumerate(hand_poses):
            # require at least 7 valid keypoints
            if (keyp != -1).sum() > 6:
                # extract only the valid ones
                valid_idxs = np.where(keyp[:, 0] != -1)
                # scale them to image coords
                xs = keyp[valid_idxs, 0].flatten() * float(W)
                ys = keyp[valid_idxs, 1].flatten() * float(H)
                # make bbox
                x0, y0 = xs.min(), ys.min()
                x1, y1 = xs.max(), ys.max()
                bboxes.append([x0, y0, x1, y1])
                # decide left (0) vs right (1)
                #  — here: even indexes are left, odd are right
                is_right.append(1 if (i % 2) else 0)

        if len(bboxes) == 0:
            return pose_img

        # stack into numpy arrays as before
        boxes = np.stack(bboxes)           # shape (num_valid_hands, 4)
        right = np.stack(is_right)        # shape (num_valid_hands,)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, raw_img_cv2, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        img_size = None

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Collect all verts and cams to list
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_hand = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_hand)

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view_diff = self.renderer.render_rgba_multiple_diff(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            processed_frame = (cam_view_diff * 255).astype(np.uint8)[:, :, ::-1]
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            processed_frame = Image.fromarray(processed_frame)
            processed_frame = self._overlay_images(pose_img, processed_frame)
            return processed_frame

        return pose_img


def overlay_images(base_image, overlay_image):
    # Convert images to RGB and then to numpy arrays
    base_image = base_image.convert("RGB")
    overlay_image = overlay_image.convert("RGB")

    base_array = np.array(base_image)
    overlay_array = np.array(overlay_image)

    # Create a mask where overlay_image is not black ([0, 0, 0])
    mask = np.all(overlay_array != [0, 0, 0], axis=-1)

    # Use the mask to replace the corresponding pixels in base_array
    base_array[mask] = overlay_array[mask]

    return Image.fromarray(base_array)