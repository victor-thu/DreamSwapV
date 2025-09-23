import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import torch

def random_mask_augmentation_dilate(
        image,
        mask,
        use_random_mask_aug=True,
        mask_color=(128, 128, 128),
        state=None
    ):
        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("L")
        # 转为 NumPy & bool mask
        image_array = np.array(image)
        mask = np.array(mask) > 128
        h, w = mask.shape
        assert image_array.shape[0] == h and image_array.shape[1] == w

        # 恢复 PyTorch RNG
        if state is not None:
            torch.set_rng_state(state)

        if use_random_mask_aug:
            if torch.rand(()) <= 0:
                # bounding box 分支
                coords = np.argwhere(mask)
                if coords.size > 0:
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0)
                    new_mask = np.zeros_like(mask)
                    new_mask[y0:y1+1, x0:x1+1] = True
                else:
                    new_mask = mask.copy()
            else:
                # 非 boundingbox 分支：保留原始网格块，并行处理 膨胀 和 特殊形状
                # 网格划分
                kh = int(torch.randint(10, 40, (1,)).item())
                kw = int(torch.randint(10, 40, (1,)).item())
                row_idx = np.linspace(0, h, kh+1, dtype=int)
                col_idx = np.linspace(0, w, kw+1, dtype=int)

                # 找到初始 filled_blocks
                filled_blocks = []
                for i in range(kh):
                    rs, re = row_idx[i], row_idx[i+1]
                    for j in range(kw):
                        cs, ce = col_idx[j], col_idx[j+1]
                        if np.any(mask[rs:re, cs:ce]):
                            filled_blocks.append((i, j))

                total_blocks = len(filled_blocks)
                # 构造基准网格块 mask：确保原始块完整保留
                base_block_mask = np.zeros_like(mask, dtype=bool)
                for (i, j) in filled_blocks:
                    rs, re = row_idx[i], row_idx[i+1]
                    cs, ce = col_idx[j], col_idx[j+1]
                    base_block_mask[rs:re, cs:ce] = True

                new_mask = base_block_mask.copy()

                if total_blocks > 0:
                    # 分别决定是否进行膨胀和贴形状（独立）
                    dilate_flag = torch.bernoulli(torch.full((), 0.5)).item() > 0
                    shape_flag = torch.bernoulli(torch.full((), 0.5)).item() > 0

                    # 膨胀逻辑：保证格子连通，无中间空洞
                    if dilate_flag:
                        iters = max(1, int(total_blocks * 0.2))
                        # 构建块级网格
                        grid = np.zeros((kh, kw), dtype=bool)
                        for i, j in filled_blocks:
                            grid[i, j] = True
                        # 随机扩张
                        block_set = set(filled_blocks)
                        neighbors = {
                            (ni, nj)
                            for (i, j) in block_set
                            for di in (-1, 0, 1)
                            for dj in (-1, 0, 1)
                            if not (di == 0 and dj == 0)
                            for ni, nj in [(i + di, j + dj)]
                            if 0 <= ni < kh and 0 <= nj < kw and (ni, nj) not in block_set
                        }
                        for _ in range(iters):
                            if not neighbors:
                                break
                            ni, nj = list(neighbors)[int(torch.randint(0, len(neighbors), (1,)).item())]
                            block_set.add((ni, nj))
                            neighbors.remove((ni, nj))
                            neighbors |= {
                                (nni, nnj)
                                for di in (-1, 0, 1)
                                for dj in (-1, 0, 1)
                                for nni, nnj in [(ni + di, nj + dj)]
                                if 0 <= nni < kh and 0 <= nnj < kw and (nni, nnj) not in block_set
                            }
                        # 更新 grid
                        for i, j in block_set:
                            grid[i, j] = True
                        # 行内填充空洞
                        for i in range(kh):
                            cols = np.where(grid[i])[0]
                            if cols.size > 1:
                                grid[i, cols.min():cols.max() + 1] = True
                        # 映射回像素
                        for i, j in np.argwhere(grid):
                            new_mask[row_idx[i]:row_idx[i+1], col_idx[j]:col_idx[j+1]] = True

                    # 特殊形状逻辑：网格级，基于类型精确构造circle/triangle/rectangle
                    if shape_flag:
                        shape_size = max(1, int(total_blocks * 0.2))
                        # 选取一个原始邻接块
                        b_i, b_j = filled_blocks[int(torch.randint(0, total_blocks, (1,)).item())]
                        # 找到一个空邻居作为种子
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = b_i+di, b_j+dj
                            if 0 <= ni < kh and 0 <= nj < kw and (ni, nj) not in filled_blocks:
                                seed_i, seed_j = ni, nj
                                break
                        else:
                            seed_i, seed_j = b_i, b_j

                        shape_type = ['circle','triangle','rectangle'][int(torch.randint(0,3,(1,)).item())]
                        shape_grid = np.zeros((kh, kw), dtype=bool)

                        if shape_type == 'circle':
                            # 圆形：半径按shape_size近似计算
                            rad = int(np.ceil(np.sqrt(shape_size/np.pi)))
                            for i in range(kh):
                                for j in range(kw):
                                    if (i-seed_i)**2 + (j-seed_j)**2 <= rad**2:
                                        shape_grid[i, j] = True
                        elif shape_type == 'triangle':
                            # 随机选择等边或直角三角形
                            tri_equ = torch.rand(()) < 0.5
                            if tri_equ:
                                # 等边三角：高按shape_size近似
                                height = int(np.ceil(np.sqrt(4*shape_size/np.sqrt(3))))
                                for di in range(height):
                                    row = seed_i + di
                                    if 0 <= row < kh:
                                        width = int((1 - di/height) * height)
                                        start = seed_j - width//2
                                        shape_grid[row, max(0,start):min(kw, start+width)] = True
                            else:
                                # 直角三角：两直边长度按shape_size估计
                                leg = int(np.ceil(np.sqrt(2*shape_size)))
                                for di in range(leg):
                                    for dj in range(leg-di):
                                        i = seed_i + di
                                        j = seed_j + dj
                                        if 0 <= i < kh and 0 <= j < kw:
                                            shape_grid[i, j] = True
                        else:
                            # 矩形：随机长宽组合
                            w_blocks = int(np.ceil(np.sqrt(shape_size)))
                            h_blocks = int(np.ceil(shape_size / w_blocks))
                            if torch.rand(()) < 0.5:
                                w_blocks, h_blocks = h_blocks, w_blocks
                            for di in range(h_blocks):
                                for dj in range(w_blocks):
                                    i = seed_i + di
                                    j = seed_j + dj
                                    if 0 <= i < kh and 0 <= j < kw:
                                        shape_grid[i, j] = True

                        # 映射shape_grid至像素级mask
                        for i, j in np.argwhere(shape_grid):
                            new_mask[row_idx[i]:row_idx[i+1], col_idx[j]:col_idx[j+1]] = True
        else:
            new_mask = mask.copy()

        # 合成 agnostic 图
        mask_color = np.array((128, 128, 128), dtype=np.uint8).reshape(1, 1, 3)
        agnostic = np.where(
            new_mask[..., None],
            mask_color,
            image_array
        ).astype(np.uint8)

        mask = Image.fromarray((new_mask*255).astype(np.uint8))
        agnostic = Image.fromarray(agnostic)

        mask.save("mask_aug.png")
        agnostic.save("agnostic_aug.png")

random_mask_augmentation_dilate("/mnt/bn/aigc-algorithm-group/weitao/humanvid_data/5837589-hd_1920_1080_24fps/5837589-hd_1920_1080_24fps_video/00001.jpg", 
                       "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/tracking_results/5837589-hd_1920_1080_24fps/5837589-hd_1920_1080_24fps_masks/Person/00000.png")