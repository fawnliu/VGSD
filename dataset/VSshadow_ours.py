import os
import os.path

import torch.utils.data as data
from PIL import Image
import random
import torch
import numpy as np
import pdb 

from glob import glob

def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

# return image triple pairs in video and return single image
class CrossPairwiseImg(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, 
                 reflection_root=None):
        self.img_root, self.video_root = self.split_root(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform

        self.input_folder = 'JPEGImages'
        self.label_folder = 'SegmentationClassPNG'
        self.img_ext = '.jpg'
        self.label_ext = '.png'

        print(self.video_root)

        self.num_video_frame = 0
        # get all frames from video datasets
        self.videoImg_list = self.generateImgFromVideo(self.video_root)
        print('Total video frames is {}.'.format(self.num_video_frame))
        # get all frames from image datasets
        if len(self.img_root) > 0:
            self.singleImg_list = self.generateImgFromSingle(self.img_root)
            print('Total single image frames is {}.'.format(len(self.singleImg_list)))
        
        self.reflection_root = reflection_root
        print('reflection_root: ', reflection_root)
        print(f"{self.video_root[0][0]} has {len(os.listdir(self.video_root[0][0]))} videos.")
        
        
    def get_ref(self, img_path):
        video = img_path.split('/')[-3]
        img_name = img_path.split('/')[-1]

        path = os.path.join(self.reflection_root, video, img_name)
        try:
            ref = Image.open(path).convert('RGB')
        except:
            # ref = Image.open(path.replace('jpg', 'png')).convert('RGB')
            ref = Image.open(path.replace('.jpg', '_fake_Rs_03.png')).convert('RGB')

        return ref 


    def __getitem__(self, index):
        manual_random = random.random()  # random for transformation
        # pair in video
        exemplar_path, exemplar_gt_path, videoStartIndex, videoLength = self.videoImg_list[index]  # exemplar
        # sample from same video
        query_index = np.random.randint(videoStartIndex, videoStartIndex + videoLength)
        if query_index == index:
            query_index = np.random.randint(videoStartIndex, videoStartIndex + videoLength)

        # # sample from different video
        # while True:
        #     other_index = np.random.randint(0, self.__len__())
        #     if other_index < videoStartIndex or other_index > videoStartIndex + videoLength - 1:
        #         break  # find image from different video

        # print(index, query_index)
        
        other_index = index + 1
        if other_index >= videoStartIndex + videoLength - 1: # index is the last frame in video
            other_index = videoStartIndex # select the first frame in video
        ## swap
        query_index, other_index = other_index, query_index
        
        query_path, query_gt_path, videoStartIndex2, videoLength2 = self.videoImg_list[query_index]  # query
        if videoStartIndex != videoStartIndex2 or videoLength != videoLength2:
            raise TypeError('Something wrong')
        other_path, other_gt_path, videoStartIndex3, videoLength3 = self.videoImg_list[other_index]  # other
        # if videoStartIndex != videoStartIndex3 or videoLength != videoLength3:
        #     raise TypeError('Something wrong')
        # if videoStartIndex == videoStartIndex3:
        #     raise TypeError('Something wrong')
        # single image in image dataset
        if len(self.img_root) > 0:
            single_idx = np.random.randint(0, videoLength)
            single_image_path, single_gt_path = self.singleImg_list[single_idx]  # single image
        
        # read image and gt
        exemplar = Image.open(exemplar_path).convert('RGB')
        query = Image.open(query_path).convert('RGB')
        other = Image.open(other_path).convert('RGB')
        exemplar_gt = Image.open(exemplar_gt_path).convert('L')
        query_gt = Image.open(query_gt_path).convert('L')
        other_gt = Image.open(other_gt_path).convert('L')
        if len(self.img_root) > 0:
            single_image = Image.open(single_image_path).convert('RGB')
            single_gt = Image.open(single_gt_path).convert('L')

        if self.reflection_root is not None:
            exemplar_ref = self.get_ref(exemplar_path)
            query_ref = self.get_ref(query_path)
            other_ref = self.get_ref(other_path)

        # transformation
        if self.joint_transform is not None:
            if self.reflection_root is None:
                exemplar, exemplar_gt = self.joint_transform(exemplar, exemplar_gt, manual_random, None)
                query, query_gt, query_edge = self.joint_transform(query, query_gt, manual_random, None, query_edge)
                other, other_gt, other_edge = self.joint_transform(other, other_gt, None, None, other_edge)
            else:
                exemplar, exemplar_gt, exemplar_ref = self.joint_transform(exemplar, exemplar_gt, manual_random, exemplar_ref)
                query, query_gt, query_ref, query_edge = self.joint_transform(query, query_gt, manual_random, query_ref, query_edge)
                other, other_gt, other_ref, other_edge = self.joint_transform(other, other_gt, None, other_ref, other_edge)
            if len(self.img_root) > 0:
                single_image, single_gt = self.joint_transform(single_image, single_gt)
        if self.img_transform is not None:
            exemplar = self.img_transform(exemplar)
            query = self.img_transform(query)
            other = self.img_transform(other)
            if len(self.img_root) > 0:
                single_image = self.img_transform(single_image)
        if self.target_transform is not None:
            exemplar_gt = self.target_transform(exemplar_gt)
            query_gt = self.target_transform(query_gt)
            other_gt = self.target_transform(other_gt)
            if len(self.img_root) > 0:
                single_gt = self.target_transform(single_gt)
            if self.reflection_root is not None:
                exemplar_ref = self.target_transform(exemplar_ref)
                query_ref = self.target_transform(query_ref)
                other_ref = self.target_transform(other_ref)
            # exemplar_edge = self.target_transform(exemplar_edge)
            query_edge = self.target_transform(query_edge)
            other_edge = self.target_transform(other_edge)

        if len(self.img_root) > 0:
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 'query': query, 'query_gt': query_gt,
                  'other': other, 'other_gt': other_gt, 'single_image': single_image, 'single_gt': single_gt}
        else:
            sample = {'exemplar': exemplar, 'exemplar_gt': exemplar_gt, 
                      'query': query, 'query_gt': query_gt,
                      'other': other, 'other_gt': other_gt
                      }
        if self.reflection_root is not None:   
            sample['exemplar_ref'] = exemplar_ref * exemplar_gt
            sample['query_ref'] = query_ref * query_gt
            sample['other_ref'] = other_ref * other_gt   ### multi gt mask here !! 
            
        sample['exemplar_path'] = exemplar_path
        sample['query_path'] = query_path
        sample['other_path'] = other_path 

        return sample

    def generateImgFromVideo(self, root):
        imgs = []
        root = root[0]  # assume that only one video dataset
        video_list = listdirs_only(os.path.join(root[0]))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root[0], video, self.input_folder)) if f.endswith(self.img_ext)] # no ext
            img_list = self.sortImg(img_list)
            for img in img_list:
                # videoImgGt: (img, gt, video start index, video length)
                videoImgGt = (os.path.join(root[0],video,  self.input_folder, img + self.img_ext),
                        os.path.join(root[0], video, self.label_folder, img + self.label_ext), self.num_video_frame, len(img_list))
                imgs.append(videoImgGt)
            self.num_video_frame += len(img_list)

        return imgs

    def generateImgFromSingle(self, root):
        imgs = []
        for sub_root in root:
            tmp = self.generateImagePair(sub_root[0])
            imgs.extend(tmp)  # deal with image case
            print('Image number of ImageSet {} is {}.'.format(sub_root[2], len(tmp)))

        return imgs

    def generateImagePair(self, root):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, self.input_folder)) if f.endswith(self.img_ext)]
        if len(img_list) == 0:
            raise IOError('make sure the dataset path is correct')
        return [(os.path.join(root, self.input_folder, img_name + self.img_ext), os.path.join(root, self.label_folder, img_name + self.label_ext))
            for img_name in img_list]

    def sortImg(self, img_list):
        img_int_list = [int(f) for f in img_list]
        sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
        return [img_list[i] for i in sort_index]

    def split_root(self, root):
        if not isinstance(root, list):
            raise TypeError('root should be a list')
        img_root_list = []
        video_root_list = []
        for tmp in root:
            if tmp[1] == 'image':
                img_root_list.append(tmp)
            elif tmp[1] == 'video':
                video_root_list.append(tmp)
            else:
                raise TypeError('you should input video or image')
        return img_root_list, video_root_list

    def __len__(self):
        return len(self.videoImg_list)//2*2

