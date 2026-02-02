import numpy as np

def concat_images(imgs: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple environment images into a single image.
    """
    if imgs.ndim == 3:
        imgs = imgs[None]
    env_num, h, w, c = imgs.shape
    grid_size = int(np.ceil(np.sqrt(env_num)))
    # Pad with black images if needed
    if env_num < grid_size ** 2:
        padding = np.zeros((grid_size ** 2 - env_num, h, w, c), dtype=imgs.dtype)
        imgs = np.concatenate([imgs, padding], axis=0)
    # Reshape to grid and concatenate
    imgs = imgs.reshape(grid_size, grid_size, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4).reshape(grid_size * h, grid_size * w, c)
    return imgs

def blend_images_half_half(img1: np.ndarray, img2: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """
    Blend two images side-by-side or top-bottom for alignment comparison.
    
    Args:
        img1: First image, shape (h, w, c) or (h, w)
        img2: Second image, shape (h, w, c) or (h, w)
        mode: "horizontal" for left/right half, "vertical" for top/bottom half
    
    Returns:
        Blended image
    """
    # Ensure both images have matching dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Use smaller size for compatibility
    h = min(h1, h2)
    w = min(w1, w2)
    
    # Crop to same size
    if len(img1.shape) == 3:
        img1 = img1[:h, :w, :]
    else:
        img1 = img1[:h, :w]
    
    if len(img2.shape) == 3:
        img2 = img2[:h, :w, :]
    else:
        img2 = img2[:h, :w]
    
    if mode == "horizontal":
        # Left half img1, right half img2
        mid_w = w // 2
        if len(img1.shape) == 3:
            blended = np.zeros((h, w, img1.shape[2]), dtype=img1.dtype)
            blended[:, :mid_w, :] = img1[:, :mid_w, :]
            blended[:, mid_w:, :] = img2[:, mid_w:, :]
        else:
            blended = np.zeros((h, w), dtype=img1.dtype)
            blended[:, :mid_w] = img1[:, :mid_w]
            blended[:, mid_w:] = img2[:, mid_w:]
    else:  # vertical
        # Top half img1, bottom half img2
        mid_h = h // 2
        if len(img1.shape) == 3:
            blended = np.zeros((h, w, img1.shape[2]), dtype=img1.dtype)
            blended[:mid_h, :, :] = img1[:mid_h, :, :]
            blended[mid_h:, :, :] = img2[mid_h:, :, :]
        else:
            blended = np.zeros((h, w), dtype=img1.dtype)
            blended[:mid_h, :] = img1[:mid_h, :]
            blended[mid_h:, :] = img2[mid_h:, :]
    
    return blended

def blend_images_transparent(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two images with alpha transparency for alignment comparison.
    
    Args:
        img1: Bottom image, shape (h, w, c) or (h, w)
        img2: Top image, shape (h, w, c) or (h, w)
        alpha: Transparency in [0, 1]; 0.5 gives equal weight to both images
    
    Returns:
        Alpha-blended image
    """
    # Ensure both images have matching dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Use smaller size for compatibility
    h = min(h1, h2)
    w = min(w1, w2)
    
    # Crop to same size
    if len(img1.shape) == 3:
        img1 = img1[:h, :w, :]
    else:
        img1 = img1[:h, :w]
    
    if len(img2.shape) == 3:
        img2 = img2[:h, :w, :]
    else:
        img2 = img2[:h, :w]
    
    # Clamp alpha to valid range
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Convert to float for computation
    if img1.dtype == np.uint8:
        img1_float = img1.astype(np.float32) / 255.0
    else:
        img1_float = img1.astype(np.float32)
    
    if img2.dtype == np.uint8:
        img2_float = img2.astype(np.float32) / 255.0
    else:
        img2_float = img2.astype(np.float32)
    
    # Alpha blending: result = alpha * img1 + (1 - alpha) * img2 (img1 bottom, img2 top)
    blended_float = alpha * img1_float + (1 - alpha) * img2_float
    
    # Convert back to original dtype
    if img1.dtype == np.uint8:
        blended = np.clip(blended_float * 255.0, 0, 255).astype(np.uint8)
    else:
        blended = blended_float.astype(img1.dtype)
    
    return blended