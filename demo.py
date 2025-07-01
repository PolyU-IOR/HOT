import os
import numpy as np
from PIL import Image
from HOT_Solver import HOTSolver
from sklearn.preprocessing import normalize

m = n = 256         # 2D Histograms: m rows and n cols
c_Max = None

def img_2_arr(image):
    """
    Args:
        image (ImageFile): PIL Image Object

    Returns:
        L1 Normalized image
    """
    if m is None or n is None:
        print("m or n is not initialized")
        raise ValueError
    image = image.resize((m, n))
    image = np.array(image)                         # To numpy array
    image_arr = image.flatten('F')                  # [expand by column]
    image_arr = normalize([image_arr], norm='l1')
    return image_arr[0]


def generate_LP_param(img1_path: str, img2_path: str):
    # read image and normalie
    global c_Max
        
    # Uniform bins
    img_1 = Image.open(img1_path).convert('L')              # grayscale image     
    img_2 = Image.open(img2_path).convert('L')              # grayscale image  
    supplier = img_2_arr(img_1)                             # normalized grayscale image
    customer = img_2_arr(img_2)                             # normalized grayscale image
    
    # OT parameter
    c1 = np.repeat(np.arange(0, m)[np.newaxis, :], m, axis=0)
    c1 = np.square(c1 - np.arange(0, m)[:, np.newaxis])
    c1 = c1.flatten(order='C')
    c1 = np.tile(c1, n)
    assert len(c1) == m * m * n, "c1 initialization error"
    
    c2 = np.repeat(np.arange(0, n)[np.newaxis, :], n, axis=0)
    c2 = np.square(c2 - np.arange(0, n)[:, np.newaxis])
    c2 = c2.flatten(order='F')
    c2 = np.repeat(c2, m)
    assert len(c2) == n * n * m, "c2 initialization error"
    
    c = np.concatenate([c1, c2])
    
    c_Max = np.max(c1) + np.max(c2)
    c = c / c_Max

    b_bar = np.concatenate([np.zeros(m * n), supplier, customer])
    b = b_bar
    
    return b, c


if __name__ == "__main__":
    img1_path = "./Dataset/Shapes/pic/6.png"
    img2_path = "./Dataset/Shapes/pic/8.png"
    
    m = n = 256
    
    b, c = generate_LP_param(img1_path, img2_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    optimizer = HOTSolver(
        b = b,
        c = c,
        m = m,
        n = n,
        cMax = c_Max,
        max_iters = 1E6,
        tolerance = 1E-6,
        check_freq = 100,
        sigma = None,       # set to 'None' to use our default initialization
        adjust_sigma = True,
        logging = True,
        dtype = None,       # by default is torch.float64
        device = None       # by defaut, [device = 'cuda' if torch.cuda.is_available() else 'cpu']
    )
    
    log = optimizer.optimize()
    
    optimizer.compute_feaserr()
    
    print(f"Finish. Prime feasibility error is {log['pr_feaserr']}, prime objective is {log['pr_obj']}.")
    
    # ---------------- logged items ---------------- #
    # log['iter']       # running iterations 
    # log['pr_obj']     # prime optimal objective
    # log['dual_obj']   # dual optimal objective
    # log['gap']        # relative prime-dual gap       
    # log['kkt_error']  # KKT error
    # log['pr_feaserr'] # prime feasibility error