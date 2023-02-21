
'''
Part of this code is extracted from https://github.com/TengdaHan/DPC
'''
from joblib import delayed, Parallel
import os 
import sys 
import glob 
from tqdm import tqdm 
import cv2

def extract_video_opencv(v_path, f_root):
    '''v_path: single video path;
       f_root: root to store frames'''
    
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_name)
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(v_path, 'not successfully loaded, drop ..'); return
    width = int(width * 0.5)
    height = int(height * 0.5)
    new_dim = (width, height)
    print(v_name + " " + str(nb_frames) + " " + str(int(vidcap.get(cv2.CAP_PROP_FPS))))
    # success, image = vidcap.read()
    # count = 0
    # while success:
    #     image = cv2.resize(image, new_dim)
    #     cv2.imwrite(os.path.join(out_dir, f'{str(count).zfill(6)}.jpg'), image)
    #     success, image = vidcap.read()
    #     count += 1
    # if nb_frames > count:
    #     print('/'.join(out_dir.split('/')[-2::]), 'NOT extracted successfully: %df/%df' % (count, nb_frames))
    vidcap.release()


def main_adl(v_root, f_root):
    print('extracting adl frames ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % f_root)
    
    if not os.path.exists(f_root): os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, '*.MP4'))
    for video in v_act_root:
        extract_video_opencv(video, f_root)
    # Parallel(n_jobs=32)(delayed(extract_video_opencv)(p, f_root) for p in tqdm(v_act_root, total=len(v_act_root)))

if __name__ == '__main__':
    main_adl(v_root='/datasets/ADL/videos/', f_root='/datasets/ADL/rgb_frames')