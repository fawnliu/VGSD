import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from config import ViSha_test_root
from misc import check_mkdir

from networks.VGD_reflection import VGD_Network 

from dataset.VSshadow_ours import listdirs_only
import argparse
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = {
    'scale': 416,
    'test_adjacent': 1,
    'input_folder': 'JPEGImages',
    'label_folder': 'SegmentationClassPNG'
}

img_transform = transforms.Compose([
    transforms.Resize((config['scale'], config['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

root = ViSha_test_root[0]
print('root: ', root)

to_pil = transforms.ToPILImage()

import pdb 
parser = argparse.ArgumentParser()
parser.add_argument("-pred", "--prediction", type=str, default=None)  #results/
parser.add_argument("-exp", "--exp", type=str, default="VMD_ours")
args = parser.parse_args()

def save_reflection(image_name, pred, d_dir, size):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = np.clip(predict_np, 0, 1)
    predict_np = np.transpose(predict_np, (1, 2, 0)) * 255.
    im = Image.fromarray((predict_np.astype(np.uint8))).convert('RGB')

    imo = im.resize(size, resample=Image.BILINEAR)

    imo.save(os.path.join(d_dir, image_name + '.png'))


def main():
    net = VGD_Network().cuda()

    print(args.prediction)
    print(args.exp)

    # checkpoint = os.path.join(args.exp, 'best.pth')
    checkpoint = args.exp 
    print(checkpoint)
    check_point = torch.load(checkpoint)
    msg = net.load_state_dict(check_point['model'], strict=False)
    print(msg)

    import time 
    all_time = 0
    index = 0

    net.eval()
    with torch.no_grad():
        video_list = listdirs_only(os.path.join(root))
        # print(video_list)
        video_list = sorted(video_list)
        for video in tqdm(video_list):
            # all images
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, config['input_folder'])) 
                        if f.endswith('.jpg')]
            # need evaluation images
            img_eval_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, video, config['label_folder'])) 
                             if f.endswith('.png')]

            img_eval_list = sortImg(img_eval_list)
            for exemplar_idx, exemplar_name in enumerate(img_eval_list):
                query_idx_list = getAdjacentIndex(exemplar_idx, 0, len(img_list), config['test_adjacent'])

                for query_idx in query_idx_list:
                    index += 1 
                    exemplar = Image.open(os.path.join(root, video, config['input_folder'], exemplar_name + '.jpg')).convert('RGB')
                    w, h = exemplar.size
                    query = Image.open(os.path.join(root, video, config['input_folder'], img_list[query_idx] + '.jpg')).convert('RGB')
                    exemplar_tensor = img_transform(exemplar).unsqueeze(0).cuda()
                    query_tensor = img_transform(query).unsqueeze(0).cuda()
                    start = time.time()
                    exemplar_final, exemplar_ref, exemplar_pre = net(exemplar_tensor, query_tensor, query_tensor)
                    all_time += time.time() - start 

                    # exemplar_final = exemplar_pre  #### NOTE

                    res = (exemplar_final.data > 0).to(torch.float32).squeeze(0)
                    # res = torch.sigmoid(exemplar_final.squeeze())
                    prediction = np.array(
                        transforms.Resize((h, w))(to_pil(res.cpu())))

                    check_mkdir(os.path.join(args.prediction, 'pred', video))
                    save_name = f"{exemplar_name}.png"
                    Image.fromarray(prediction).save(os.path.join(args.prediction, 'pred', video, save_name))

                    # print(os.path.join(args.prediction, 'reflection', video), '---')

                    # # ## save reflection 
                    # check_mkdir(os.path.join(args.prediction, 'reflection', video))
                    # save_reflection(exemplar_name, exemplar_ref, os.path.join(args.prediction, 'reflection', video), (w, h))



def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


def getAdjacentIndex(current_index, start_index, video_length, adjacent_length):
    if current_index + adjacent_length < start_index + video_length:
        query_index_list = [current_index+i+1 for i in range(adjacent_length)]
    else:
        query_index_list = [current_index-i-1 for i in range(adjacent_length)]
    return query_index_list

if __name__ == '__main__':
    main()
