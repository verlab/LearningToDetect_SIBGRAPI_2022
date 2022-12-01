import numpy as np
import cv2
import os, glob, argparse, tqdm
import h5py

import random
import sys

import math

import torch.nn.functional as F
import modules.augmentation as aug
import torch

from scipy.spatial import Delaunay

import uuid

paths = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../')
if not os.path.exists(paths): 
    raise RuntimeError('Invalid path for descriptor tools: ' + paths)
else:
    print("ok path:", paths)
sys.path.insert(0, paths)

# Load DEAL
from modules.utils import DEAL #include the modules on your path
deal_net_path = 'models/newdata-DEAL-big.pth'
deal = DEAL(deal_net_path, sift = False) # Create the descriptor and load arctecture

from models.cnn_wrapper.aslfeat import ASLFeatNet
obj = ASLFeatNet(None, False)
obj.init()

np.set_printoptions(suppress=True)
#rng.seed(1)

args = None

def _run(data, obj):
    obj.netConfig
    obj.sess
    max_dim = max(data.shape[0], data.shape[1])
    #if len(data.shape) == 2:
        #data = np.expand_dims(data, 0)
    h, w, _ = data.shape
    
    feed_dict = {"input:0": np.expand_dims(data, 0)}
    returns = obj.sess.run(obj.endpoints, feed_dict=feed_dict)
    feature_maps = returns['feature_maps']
    keypoints_asl = returns['keypoints']
    descs = returns['descs'][0]

    idx = 0
    d = np.zeros((h, w, 128), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            d[y][x] = descs[idx]
            idx+=1
    
    # retornando o ultimo feature map e os descritores no shape da imagem de entrada
    feature_map = feature_maps[2]
    return feature_map, d, keypoints_asl

# grayscale nonmaxima
def nonMaxima(img, n=13): # n = window size
    mid = int(n/2)
    h, w = img.shape
    for i in range(h-n-1):
        a = i+n
        for j in range(w-n-1):
            b = j+n
            p = img[mid+i][mid+j]
            if p == 0:
                continue
            t = False
            for k in range(i, a):
                for l in range(j, b):
                    if p < img[k][l]:
                        img[mid+i][mid+j] = 0
                        t = True
                        break
                if t == True:
                    break

    return img

#### Extracting Autocorrelation of descriptors 

def descriptorSSD(descMat, x, y, w=3):
    f1 = descMat[y][x]
    H, W = descMat.shape[0], descMat.shape[1]
    m = int(w/2)
    s = 0
    for u in range(-m, m+1):
        for v in range(-m, m+1):
            if y+u < 0 or x+v < 0 or y+u >= H or x+v >= W:
                continue

            if descMat[y+u][x+v] is None:
                continue
            f2 = descMat[y+u][x+v]
            s += np.linalg.norm(f1 - f2)
            
    return s

def autocorrelation(desc, harris_th=5):
    h = desc.shape[0]
    w = desc.shape[1]
    responseImg = np.zeros((h, w), dtype=np.float32)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #desc = torch.Tensor(desc).to(device)
    print("init SSD")
    # pegando a autorrelacao de cada descritor    
    for y in range(h):
        for x in range(w):
            responseImg[y][x] = descriptorSSD(desc, x, y, w=5)
    
    print("end SSD")
    responseImg = np.asarray(responseImg)
    # fazendo non maxima supression na respostas dos descritores
    winSize = 5
    responseImg = cv2.GaussianBlur(responseImg, (3, 3), 0)
    
    print("autocorrelation estatistics before: ", responseImg.min(), responseImg.max(), responseImg.std())

    h, w = responseImg.shape
    for y in range(h):
        for x in range(w):
            responseImg[y][x] = 0 if responseImg[y][x] < harris_th else responseImg[y][x]
    
    #print("autocorrelation estatistics after: ", responseImg.min(), responseImg.max(), responseImg.std())
    responseImg = nonMaxima(responseImg, winSize)
    
    pixels = []
    arr = []
    for y in range(h):
        for x in range(w):
            if responseImg[y][x] != 0:
                pixels.append([x, y])
                arr.append(desc[y][x])
    
    return arr, pixels

def detect(img):
    # descritors
    feature_map, desc_map, keypoints_asl = _run(img, obj)

    desc, new_kps_x_tosave = autocorrelation(desc_map)

    return new_kps_x_tosave, desc, feature_map

def detect2(img):
    # descritors
    feature_map, desc_map, keypoints_asl = _run(img, obj)

    return feature_map, desc_map, keypoints_asl


def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input path containing images"
    , required=True) 
    parser.add_argument("-o", "--output", help="Output path where results will be saved."
    , required=True) 
    parser.add_argument("-n", "--ntotal", help="Number of images to sample.", type=int
    , required=False, default = 10) 
    parser.add_argument("-b", "--begin", help="Folder id to initiate", type=int
    , required=True, default = 0)
    parser.add_argument("-e", "--end", help="Folder id to finalize", type=int
    , required=True, default = -1) 

    parser.add_argument("--createh5", help="Create H5 dataset instead of building raw dataset"
    , action = 'store_true') 

    args = parser.parse_args()
    return args

def check_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)

def create_h5dataset(raw_dataset_path, out_path):
    dataset = h5py.File(out_path, "w")
    dataset.create_group('imgs')
    dataset.create_group('gts')

    cont = 0
    num_kps_sum = 0
    
    for pair in tqdm.tqdm(glob.glob(raw_dataset_path + '/*0-rgb.png')):
        pair = pair.replace('-rgb','')
        pair_id, ext = os.path.splitext(pair)
        pair_id = pair_id[:-1]
        try:
            img = cv2.imread(pair_id +'0-rgb.png')
            h,w = img.shape[0], img.shape[1]
            gt = np.load(pair_id + '2.npy')
            num_kps = gt.sum()/255
            num_kps_sum += num_kps            
            # base image
            imgB = cv2.imread(pair_id +'3-rgb.png')
            gtB = np.load(pair_id + '5.npy')

        except:
            print("Skiping image by error!")
            continue
        cont+=1
        pair_id = os.path.basename(pair_id)
        #print('id:', pair_id)
        dataset['imgs'].create_dataset(pair_id+'00__H', data = np.asarray([h]), compression="gzip", compression_opts=9)
        dataset['imgs'].create_dataset(pair_id+'00__W', data = np.asarray([w]), compression="gzip", compression_opts=9)
        dataset['imgs'].create_dataset(pair_id+'00__img', data = img, compression="gzip", compression_opts=9)
        
        dataset['gts'].create_dataset(pair_id+'00__', data = gt, compression="gzip", compression_opts=9)
        dataset['imgs'].create_dataset(pair_id+'00__imgB', data = imgB, compression="gzip", compression_opts=9)
        dataset['gts'].create_dataset(pair_id+'00__B', data = gtB, compression="gzip", compression_opts=9)
    
    print("Final dataset images len: ", cont)

# using covex hull to select region where there are commum repeatable points in the pair images
def remove_out_roi(gt, img):
    kps = []
    h, w = gt.shape
    mask = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            if gt[y][x] > 0:
                kps.append([x, y])
    hull = Delaunay(kps)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if hull.find_simplex([[x,y]]) >= 0:
                mask[y][x] = 1
    
    mask = cv2.dilate(mask, None, iterations=6)

    return img * mask


def match_pair(imgs, Hs, augmentor, pair_id):
    imgs_copy, kps, desc, imgs_ori, kps_cv2 = [], [], [], [], []
    for i in range(len(imgs)):
        imgs[i] = imgs[i].permute(1, 2, 0).cpu().detach().numpy()*255
        imgs[i] = imgs[i].astype(np.uint8)
        imgs_ori.append(np.copy(imgs[i]))
        imgs_copy.append(np.copy(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)))
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        
        print("using aslfeat descriptor")
        _, desc_m, kps_asl = detect2(imgs[i])
        
        desc_aux = []
        kps_aux = [[int(kp[0]), int(kp[1])] for kp in kps_asl[0]]
        for kp in kps_aux:
            x, y = kp
            desc_aux.append(desc_m[y][x])
        desc_aux = np.asarray(desc_aux)

        print("Num kps:", len(kps_aux)) 
        #print("using deal descriptor")
        #desc_aux = deal.compute(imgs_copy[i], kps_cv2)
        #kps_aux = [[int(kp.pt[0]), int(kp.pt[1])] for kp in kps_aux]

        desc.append(desc_aux)
        kps.append(kps_aux)
        
        
    ## geting inliers and matches
    H1, W1 = imgs[0].shape[0], imgs[0].shape[1]
    gt_root = []
    gt_root_rr1, gt_root_rr2 = [], []
    desc1 = desc[0]
    kps1 = kps[0]
    kps1_tensor = torch.Tensor([kps[0]]).to(augmentor.device)
    ref_correct = []
    for i in range(1, len(imgs)): # pegando da segunda imagem em diante considerando a primeira como a root
        kps2_tensor = torch.Tensor([kps[i]]).to(augmentor.device)

        _, inliers1, inliers2 = aug.get_positive_corrs(kps1_tensor, kps2_tensor, Hs[i-1], augmentor)

        inliers1 = torch.squeeze(inliers1, -1).numpy()
        inliers2 = torch.squeeze(inliers2, -1).numpy()

        print('inliers len: ', inliers1.shape, inliers2.shape)

        desc2 = np.asarray(desc[i])
        kps2 = kps[i]

        ## keypoints with repeatability ##
        x_positives_rr1, x_positives_rr2 = [], []
        for id in range(len(inliers1)):
            x_positives_rr1.append(kps1[inliers1[id]])
            x_positives_rr2.append(kps2[inliers2[id]])
        gt_root_rr1.append(x_positives_rr1)
        gt_root_rr2.append(x_positives_rr2)

        print('Lens descs: ', len(desc1), len(desc2))

        bf = cv2.BFMatcher()
        matches1 = bf.knnMatch(desc1, desc2, k=2)
        matches2 = bf.knnMatch(desc2, desc1, k=2)

        if len(matches1) == 0 or len(matches2) == 0:
            return
    
        ### matching and getting samples
        matchesMask = [[0,0] for i in range(len(matches1))]
        x_positives1 = []
        kps_cv2_filter1, kps_cv2_filter2 = [], []
        ref_correct_img2 = [[[-1,-1] for _ in range(W1)] for _ in range(H1)]

        for k,(m,n) in enumerate(matches1):
            id1 = m.queryIdx
            id2 = m.trainIdx

            pos1 = np.where(inliers1 == id1)[0]
            pos2 = np.where(inliers2 == id2)[0]
            #print('id desc1:', id1, ' id desc2: ', id2)
            #print('positions: ', pos1, ' <> ', pos2)

            if len(pos1) == 0 or len(pos2) == 0:
                continue
            if pos1[0] == pos2[0]:
                if m.distance < 0.80*n.distance and matches2[id2][0].trainIdx == k: # if ratio test and cross validation passes
                    matchesMask[k]=[1,0]
                    x_positives1.append(kps1[id1])
                    #x_positives2.append(kps2[id2])
                    #kps_cv2_filter1.append(kps_cv2[0][id1])
                    #kps_cv2_filter2.append(kps_cv2[i][id2])
                    x , y = kps1[id1]
                    x2 , y2 = kps2[id2]
                    ref_correct_img2[y][x] = [x2, y2]

        ref_correct_img2 = np.asarray(ref_correct_img2)
        ref_correct.append(ref_correct_img2)

        draw_params = dict(matchColor = (0,255,0),
                        matchesMask = matchesMask,
                        flags = 2)
        
        # imgMatching = cv2.drawMatchesKnn(imgs_ori[0], kps_cv2[0] , imgs_ori[i], kps_cv2[i], matches1, None, **draw_params)
        # cv2.imwrite(str(i)+'_match.png', imgMatching)
        # gt1 = cv2.drawKeypoints(imgs_ori[0], kps_cv2_filter1, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # cv2.imwrite('0'+str(i)+"_pseudo-gt.png", gt1)  
        # gt2 = cv2.drawKeypoints(imgs_ori[i], kps_cv2_filter2, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # cv2.imwrite(str(i)+"_pseudo-gt.png", gt2)  

        print('acertou:', len(x_positives1))
        if len(x_positives1) < 32 or len(x_positives1) > 450:
            print("small or big number of correct matches")
            return
        
        gt_img = np.zeros((H1, W1), dtype=np.float32)
        for kp in x_positives1:
            gt_img[kp[1]][kp[0]] = 1.0
        
        gt_root.append(gt_img)

    gt_root_final = gt_root[0]
    gt_img1_2 = np.zeros((H1, W1), dtype=np.float32)
    gt_img1_3 = np.zeros((H1, W1), dtype=np.float32)
    gt_img2 = np.zeros((H1, W1), dtype=np.float32)
    gt_img3 = np.zeros((H1, W1), dtype=np.float32)

    for y in range(H1):
        for x in range(W1):
            if gt_root[1][y][x] == 1:
                gt_img1_3[y][x] = 1 if gt_img1_3[y][x] == 0 else 2
                x_c, y_c = ref_correct[1][y][x]
                gt_img3[y_c][x_c] = 1 if gt_img3[y_c][x_c] == 0 else 2
            if gt_root_final[y][x] == 1:
                gt_img1_2[y][x] = 1 if gt_img1_2[y][x] == 0 else 2
                x_b, y_b = ref_correct[0][y][x]
                gt_img2[y_b][x_b] = 1 if gt_img2[y_b][x_b] == 0 else 2

                for k in range(-1, 2):
                    for j in range(-1, 2):
                        y_, x_ = y+k, x+j
                        if gt_root[1][y_][x_] == 1:
                            gt_img1_2[y][x] = 1 if gt_img1_2[y][x] == 0 else 2
                            gt_img1_3[y][x] = 1 if gt_img1_3[y][x] == 0 else 2
                            x_b, y_b = ref_correct[0][y][x]
                            gt_img2[y_b][x_b] = 1 if gt_img2[y_b][x_b] == 0 else 2
                            x_c, y_c = ref_correct[1][y_][x_]
                            gt_img3[y_c][x_c] = 1 if gt_img3[y_c][x_c] == 0 else 2
    
    print('final root gt initial sum 1:', gt_img1_2.sum(), gt_img1_3.sum(), gt_img2.sum(), gt_img3.sum())
    
    final_match_number = gt_img3.sum()-gt_root[1].sum()
    if final_match_number < 32 or final_match_number > 400:
        print("Small or big number of final matches...")
        return
    gt_img1_2 /= 2.0
    gt_img1_3 /= 2.0
    gt_img2 /= 2.0
    gt_img3 /= 2.0
    
    print('final root gt initial sum:', gt_img1_2.sum(), gt_img1_3.sum(), gt_img2.sum(), gt_img3.sum())
    # equilibrando o número de keypoints do ground truth para ficar com 100 de somatório final
    missing1 = 100 - gt_img1_2.sum()
    missing2 = 100 - gt_img1_3.sum()
    if missing1 > 0:
        for kpi in range(len(gt_root_rr1[0])):
            kp1 = gt_root_rr1[0][kpi]
            kp2 = gt_root_rr2[0][kpi]

            if gt_img1_2[kp1[1]][kp1[0]] == 0:
                gt_img1_2[kp1[1]][kp1[0]] = 0.25
                gt_img2[kp2[1]][kp2[0]] = 0.25
                missing1 -= 0.25
            if missing1 <= 0:
                break
    
    if missing2 > 0:
        for kpi in range(len(gt_root_rr1[1])):
            kp1 = gt_root_rr1[1][kpi]
            kp2 = gt_root_rr2[1][kpi]

            if gt_img1_3[kp1[1]][kp1[0]] == 0:
                gt_img1_3[kp1[1]][kp1[0]] = 0.25
                gt_img3[kp2[1]][kp2[0]] = 0.25
                missing2 -= 0.25
            if missing2 <= 0:
                break
    print('final root gt initial sum:', gt_img1_2.sum(), gt_img1_3.sum(), gt_img2.sum(), gt_img3.sum())

    print("saving files to folder:") 

    ## Salvando arquivos
    imgs_copy_00 = remove_out_roi(gt_img1_2, imgs_copy[0])
    gt_img1_2 = (gt_img1_2/gt_img1_2.max()*255).astype(np.uint8) # pushing max to 255

    imgs_copy_01 = remove_out_roi(gt_img1_3, imgs_copy[0])
    gt_img1_3 = (gt_img1_3/gt_img1_3.max()*255).astype(np.uint8) # pushing max to 255

    imgs_copy[1] = remove_out_roi(gt_img2, imgs_copy[1])
    gt_img2 = (gt_img2/gt_img2.max()*255).astype(np.uint8) # pushing max to 255
    cv2.imwrite(args.output + '/' + pair_id + '_1__0-rgb.png', imgs_copy[1])
    np.save(args.output + '/' + pair_id + '_1__2.npy', gt_img2)
    cv2.imwrite(args.output + '/' + pair_id + '_1__3-rgb.png', imgs_copy_00)
    np.save(args.output + '/' + pair_id + '_1__5.npy', gt_img1_2)   
    
    imgs_copy[2] = remove_out_roi(gt_img3, imgs_copy[2])
    gt_img3 = (gt_img3/gt_img3.max()*255).astype(np.uint8) # pushing max to 255
    cv2.imwrite(args.output + '/' + pair_id + '_2__0-rgb.png', imgs_copy[2])
    np.save(args.output + '/' + pair_id + '_2__2.npy', gt_img3)
    cv2.imwrite(args.output + '/' + pair_id + '_2__3-rgb.png', imgs_copy_01)
    np.save(args.output + '/' + pair_id + '_2__5.npy', gt_img1_3)

    # print("Salvando arquivo")

    # cv2.imwrite("pseudo-gt_final1-2.png", cv2.GaussianBlur(gt_img1_2, (3, 3), 0))  
    # cv2.imwrite("pseudo-gt_final1-3.png", cv2.GaussianBlur(gt_img1_3, (3, 3), 0))  
    # cv2.imwrite("pseudo-gt_final2.png", cv2.GaussianBlur(gt_img2, (3, 3), 0))  
    # cv2.imwrite("pseudo-gt_final3.png", cv2.GaussianBlur(gt_img3, (3, 3), 0))
    
    

    # tensor_board_print(imgs_copy_00, gt_img1_2, pair_id+'_0')
    # tensor_board_print(imgs_copy_01, gt_img1_3, pair_id+'_1')
    # tensor_board_print(imgs_copy[1], gt_img2, pair_id+'_2')
    # tensor_board_print(imgs_copy[2], gt_img3, pair_id+'_3')

def main():
    global args
    args = parseArg()
    begin = args.begin
    end = args.end
    
    if begin > end:
        print("Error begin > end")
        return

    n_retetition = args.ntotal

    cond = args.createh5
    
    print(os.path.abspath(args.output))

    if not os.path.exists(args.output) and not cond: 
        print("error path")
        return
        

    if cond:
        create_h5dataset(raw_dataset_path = args.input, out_path = args.output)

    else:
        #print("input:", args.input + "/DATASET*")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device =  "cpu"
    
        augmentor = aug.AugmentationPipe(device = device,  
                                    img_dir = args.input+"*.jpg",#'/code/other/*.jpg',
                                    max_num_imgs = end,
                                    start_num_imgs = begin,
                                    num_test_imgs = 1,
                                    out_resolution = (500, 400), 
                                    batch_size = 4
                                    )

        difficulty = 0.2
        
        check_dir(args.output)
        for k in range(n_retetition):
            imgs1, imgs2, imgs3, Hs, Hs2 = aug.make_batch_sfm(augmentor, difficulty)
            
            for i in tqdm.tqdm(range(len(imgs1))):
                idx = random.randint(8, 30)
                pair_id = str(uuid.uuid1().int)[idx-8:idx] + "_" +str(i)
                print("pair id:", pair_id)
                match_pair([imgs1[i], imgs2[i], imgs3[i]], [Hs[i], Hs2[i]], augmentor, pair_id)
            

main()
