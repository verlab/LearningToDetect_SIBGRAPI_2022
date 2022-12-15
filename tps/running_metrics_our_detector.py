from operator import delitem
import os
import sys

from numpy.lib.function_base import kaiser

if '__file__' in vars() or '__file__' in globals():
    tps_repo_path = os.path.dirname(os.path.realpath(__file__)) + '/py-thin-plate-spline'
    if not os.path.exists(tps_repo_path):
        raise RuntimeError('TPS repository is required')
else:
    tps_repo_path = './py-thin-plate-spline'
    if not os.path.exists(tps_repo_path):
        raise RuntimeError('TPS repository is required')
                
sys.path.insert(0, tps_repo_path)

import thinplate as tps
import torch
import cv2
import glob
import tqdm
import argparse
import numpy as np
from scipy.spatial import KDTree
import random

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../runs/test_nonrigiddataset_ours_figure')

# Load DEAL
from modules.utils import DEAL #include the modules on your path
deal_net_path = 'models/newdata-DEAL-big.pth'

# Load our
sys.path.insert(0, '../../detector_master')
from models.our_detector import Our

our_detector = Our()
print("num parameters:", sum([param.nelement() for param in our_detector.parameters()]))

def write_detector(filepath, x, y):
    with open(filepath + '.our', 'w') as f:
        f.write('size, angle, x, y, octave\n')
        for i in range(len(x)):
            f.write('2.0, 0.0, %d, %d, 0\n'%(x[i], y[i]))

def write_desc(filepath, descs):
    np.savetxt(filepath + '.aslfeat', descs, delimiter=',', header=str(len(descs))+" 128", comments='')

def writeToTensorBoard(images, title, permute=True):
    c=1
    for img in images:
        #print(img.shape)
        if len(img.shape) == 2:
            img = torch.unsqueeze(img, 0)
        if len(img.shape) == 4:
            img = torch.squeeze(img)
        if permute:
            img = img.permute(2, 0, 1)

        writer.add_image(title+f"/{c}", img.to(torch.uint8), 0, dataformats='CHW')
        c+=1
    writer.close()

def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input directory for one or more datasets (use --dir for several)"
    , required=True) 
    parser.add_argument("--tps_dir", help="Input directory containing the optimized TPS params for one or more datasets (use --dir for several)"
    , required=True) 
    parser.add_argument("--model_path", help="Path to our detector trained model"
    , required=True)
    parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
    , action = 'store_true')
    parser.add_argument("--deal_desc", help="must generate descriptor of DEAL"
    , action = 'store_true')
    parser.add_argument("--save_files", help="must save keypoints and descriptors in a file in applications path"
    , action = 'store_true')
    args = parser.parse_args()
    return args

args = parseArg()
args.input = os.path.abspath(args.input)
deal_desc = args.deal_desc
save_files = args.save_files
our_detector.init_detector(args.model_path)
MAX_KPS = 1024

base_aplications = '../applications/'

if deal_desc:
    deal = DEAL(deal_net_path, sift = False) # Create the descriptor and load arctecture
else:
    deal = None

# True if is running for more than one dataset (forcing)
if args.dir:
    datasets = [d for d in glob.glob(args.input+'/*/*') if os.path.isdir(d)]
else:
    datasets = [args.input]

tps_path = args.tps_dir
#datasets = list(filter(lambda x: 'DeSurTSampled' in x or  'Kinect1' in x or 'Kinect2Sampled' in x or 'SimulationICCV' in x, datasets))
#datasets = list(filter(lambda x:  'Kinect1' in x or 'Kinect2Sampled' in x or 'SimulationICCV' in x, datasets))
datasets = list(filter(lambda x: 'Kinect1' in x, datasets))
#datasets = list(filter(lambda x: 'Kinect2Sampled' in x, datasets))
#datasets = list(filter(lambda x: 'SimulationICCV' in x, datasets))
#datasets = list(filter(lambda x: 'DeSurTSampled' in x, datasets))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Running on: ", device)

inliers_total = []
generalMMA = []
generalMS = []
generalRR = []

for dataset in datasets:
    mean_rr_dataset = 0
    MA3 = 0
    MS = 0
    print('d:', dataset)
    if len(glob.glob(dataset + '/*.csv')) == 0: raise RuntimeError('Empty dataset with no .csv file')

    targets = [os.path.splitext(t)[0] for t in glob.glob(dataset + '/*[0-9].csv')]
    master = os.path.splitext(glob.glob(dataset + '/*master.csv')[0])[0]

    loading_path = os.path.join(tps_path, *dataset.split('/')[-2:])

    ref_mask = cv2.imread(loading_path + '/' + os.path.basename(master) + '_objmask.png', 0)

    #print("Targets, martes", targets, '\n', master)
    
    # reading reference image
    ref_img = cv2.imread(master + '-rgb.png')
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_img_copy = np.copy(ref_img)
    ref_img = ref_img[..., np.newaxis]
        
    h,w = ref_img.shape[0], ref_img.shape[1]
    for y in range(h):
        for x in range(w):
            ref_img[y][x] = ref_img[y][x] if ref_mask[y][x] > 0 else 0
            #ref_img_copy[y][x] = ref_img_copy[y][x] if ref_mask[y][x] > 0 else 0

    loading_path = os.path.join(tps_path, *dataset.split('/')[-2:])

    if not os.path.exists(loading_path):
        raise RuntimeError('There is no TPS directory in ' + loading_path)
    
    ref_score_map, ref_kps, ref_descriptors = our_detector.detect(ref_img_copy, MAX_KPS, f'{master}')
    ref_kps = [cv2.KeyPoint(int(kp[0]), int(kp[1]), 2) for kp in ref_kps]
    
    print(len(ref_kps))
    if deal_desc:
        print("deal")
        ref_descriptors = deal.compute(ref_img_copy, ref_kps)
        
    ref_descriptors = [ref_descriptors[i] for i in range(len(ref_kps)) if ref_mask[int(ref_kps[i].pt[1]), int(ref_kps[i].pt[0])] > 0]
    ref_kps = [kp for kp in ref_kps if ref_mask[int(kp.pt[1]), int(kp.pt[0])] > 0] #filter by object mask

    ref_kps_cv2 = ref_kps[:]

    ref_descriptors = np.asarray(ref_descriptors)
    ref_kps = np.asarray(ref_kps)

    num_ref_kps = len(ref_kps)

    if save_files:
        x  = [kp.pt[0] for kp in ref_kps]
        y  = [kp.pt[1] for kp in ref_kps]
        write_detector(base_aplications+master.split('All_PNG/')[1], x, y)
        write_desc(base_aplications+master.split('All_PNG/')[1], ref_descriptors)

    total = 0
    for target in tqdm.tqdm(targets, desc = 'image pairs'):
        try:
            total += 1
            loading_file = loading_path + '/' + os.path.basename(target)
            theta_np = np.load(loading_file + '_theta.npy').astype(np.float32)
            ctrl_pts = np.load(loading_file + '_ctrlpts.npy').astype(np.float32)
            score = cv2.imread(loading_file + '_SSIM.png', 0) / 255.0
            tgt_mask = cv2.imread(loading_file + '_objmask.png', 0)
            target_img = cv2.imread(target + '-rgb.png')
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            target_img_copy = np.copy(target_img)
            target_img = target_img[..., np.newaxis]
            score_mask = score > 0.25
        except:
            print("Error trying to read files, skiping...")
            continue

        h,w = target_img.shape[0], target_img.shape[1]
        for y in range(h):
            for x in range(w):
                target_img[y][x] = target_img[y][x] if tgt_mask[y][x] > 0 else 0
                #target_img_copy[y][x] = target_img_copy[y][x] if tgt_mask[y][x] > 0 else 0

        target_score_map, target_kps, target_descriptors = our_detector.detect(target_img_copy, MAX_KPS, f'{target}')
        
        target_kps = [cv2.KeyPoint(int(kp[0]), int(kp[1]), 2) for kp in target_kps]
        print(len(target_kps))
        if deal_desc:
            target_descriptors = deal.compute(target_img_copy, target_kps)

        target_descriptors = [target_descriptors[i] for i in range(len(target_descriptors)) if tgt_mask[int(target_kps[i].pt[1]), int(target_kps[i].pt[0])] > 0]
        target_kps = [kp for kp in target_kps if tgt_mask[int(kp.pt[1]), int(kp.pt[0])] > 0] #filter by object mask

        target_descriptors = [target_descriptors[i] for i in range(len(target_descriptors)) if score_mask[int(target_kps[i].pt[1]), int(target_kps[i].pt[0])] == True]
        target_kps = [kp for kp in target_kps if score_mask[int(kp.pt[1]), int(kp.pt[0])] == True] #filter by score map with very low confidences

        target_kps_cv2 = target_kps[:]

        target_descriptors = np.asarray(target_descriptors)
        target_kps = np.asarray(target_kps)

        print('num kps:', len(target_kps))
        if len(target_kps) < 2:
            continue

        num_tgt_kps = len(target_kps)

        if save_files:
            x  = [kp.pt[0] for kp in target_kps]
            y  = [kp.pt[1] for kp in target_kps]
            write_detector(base_aplications+target.split('All_PNG/')[1], x, y)
            write_desc(base_aplications+target.split('All_PNG/')[1], target_descriptors)

        norm_factor = np.array(target_img.shape[:2][::-1], dtype = np.float32)
        theta = torch.tensor(theta_np, device= device)
        tgt_coords = np.array([kp.pt for kp in target_kps], dtype = np.float32) 
        warped_coords = tps.torch.tps_sparse(theta, torch.tensor(ctrl_pts, device=device), torch.tensor(tgt_coords / norm_factor, 
                                                                        device=device)).squeeze(0).cpu().numpy() * norm_factor
        tree = KDTree([kp.pt for kp in ref_kps])
        dists, idxs_ref = tree.query(warped_coords)
        px_thresh = 3.0
        gt_tgt  = np.arange(len(target_kps))[ dists < px_thresh] # Groundtruth indexes -- threshold is in pixels 
        gt_ref = idxs_ref[dists < px_thresh] 

        #filter repeated matches
        _, uidxs = np.unique(gt_ref, return_index = True)
        gt_ref = gt_ref[uidxs]
        gt_tgt = gt_tgt[uidxs]

        # create BFMatcher object
        bf = cv2.BFMatcher(crossCheck=False)
        # Match descriptors.
        matches = bf.knnMatch(ref_descriptors, target_descriptors, 2)

        #Use IDs ground-truth to plot correct and incorrect matches
        right = []
        notMatch = 0
        matchesMask = [[0,0] for i in range(len(matches))]
        matchesMask2 = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            id1 = m.queryIdx
            id2 = m.trainIdx            
        
            pos1 = np.where(gt_ref == id1)[0]
            pos2 = np.where(gt_tgt == id2)[0]
            if len(pos1) == 0 or len(pos2) == 0:
                continue
            if pos1[0] == pos2[0]:
                right.append([m])
                matchesMask[i]=[1,0]
            else:
                matchesMask2[i]=[1,0]
        
        draw_params = dict(matchColor = (0,255,0),
                        matchesMask = matchesMask,
                        flags = 2)
        draw_params2 = dict(matchColor = (255,0,0),
                        matchesMask = matchesMask2,
                        flags = 1, 
                        singlePointColor=(50, 50, 50))
        
        imgMatching = cv2.drawMatchesKnn(ref_img_copy, ref_kps , target_img_copy, target_kps, matches, None, **draw_params)
        #cv2.imwrite(f'matching1.png', imgMatching)

        out1 = cv2.drawKeypoints(ref_img_copy, ref_kps_cv2, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        out2 = cv2.drawKeypoints(target_img_copy, target_kps_cv2, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        imgMatching = cv2.drawMatchesKnn(ref_img_copy, ref_kps , target_img_copy, target_kps, matches, imgMatching, **draw_params2)
        #cv2.imwrite(f'matching2.png', imgMatching2)
        writeToTensorBoard([torch.Tensor(imgMatching), torch.Tensor(out1), torch.Tensor(out2)], f'kinect1_match_{master}_{target}')
        
        
        # Metrics calculation
        print('r:', len(right), 'gt_len:', len(gt_tgt), 'total kps', len(target_kps), 'and', len(ref_kps))
        inliers_total.append(len(gt_tgt))
        if (min(len(target_kps), len(ref_kps))) != 0:
            MS += len(right)/(min(len(target_kps), len(ref_kps)))
        if len(gt_tgt) != 0:
            MA3 += len(right)/(len(gt_tgt))
        num_gt_kps = len(gt_ref)
        rr = num_gt_kps / (1.0 * min(num_ref_kps, num_tgt_kps))
        #print("RR = ", rr)
        mean_rr_dataset += rr
            
    mean_rr_dataset = mean_rr_dataset / total
    MMA = MA3 / total
    MMS = MS / total

    generalMMA.append(MMA)
    generalMS.append(MMS)
    generalRR.append(mean_rr_dataset)

    print("MRR = ", mean_rr_dataset)
    print("MMS = ", MMS)
    print("MMA = ", MMA)

print("Inliers mean:", np.asarray(inliers_total).mean())
print(np.asarray(generalRR).mean())
print(np.asarray(generalMS).mean())
print(np.asarray(generalMMA).mean())
