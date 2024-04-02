import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, manual_random=None, ref=None):
        assert img.size == mask.size
        for t in self.transforms:
            if ref is not None:
                img, mask, ref = t(img, mask, manual_random, ref) 
            else:
                img, mask = t(img, mask, manual_random)

        if ref is not None:
            return img, mask, ref 
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, manual_random=None, ref=None):
        if manual_random is None:
            if random.random() < 0.5:
                if ref is not None:
                    return img.transpose(Image.FLIP_LEFT_RIGHT), \
                        mask.transpose(Image.FLIP_LEFT_RIGHT), ref.transpose(Image.FLIP_LEFT_RIGHT) 
                else:
                    return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            if ref is not None:
                return img, mask, ref  
            return img, mask
        else:
            if manual_random < 0.5:
                if ref is not None:
                    return img.transpose(Image.FLIP_LEFT_RIGHT), \
                        mask.transpose(Image.FLIP_LEFT_RIGHT), ref.transpose(Image.FLIP_LEFT_RIGHT) 
                else:
                    return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            if ref is not None:
                return img, mask, ref  
            return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, manual_random=None, ref=None):
        assert img.size == mask.size
        if ref is not None:
            return img.resize(self.size, Image.BILINEAR), \
                mask.resize(self.size, Image.NEAREST), ref.resize(self.size, Image.NEAREST) 
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)





# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, mask, manual_random=None, ref=None, edge=None):
#         assert img.size == mask.size

#         for t in self.transforms:
#             if ref is not None:
#                 img, mask, ref, edge = t(img, mask, manual_random, ref, edge) 
#             else:
#                 img, mask, edge = t(img, mask, manual_random, None, edge)

#         if ref is not None:
#             return img, mask, ref, edge 
#         return img, mask, edge 


# class RandomHorizontallyFlip(object):
#     def __call__(self, img, mask, manual_random=None, ref=None, edge=None):
#         if manual_random is None:
#             if random.random() < 0.5:
#                 if ref is not None:
#                     return img.transpose(Image.FLIP_LEFT_RIGHT), \
#                         mask.transpose(Image.FLIP_LEFT_RIGHT), \
#                         ref.transpose(Image.FLIP_LEFT_RIGHT), \
#                         edge.transpose(Image.FLIP_LEFT_RIGHT) 
#                 else:
#                     return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), \
#                         edge.transpose(Image.FLIP_LEFT_RIGHT) 
#             if ref is not None:
#                 return img, mask, ref, edge 
#             return img, mask, edge 
#         else:
#             if manual_random < 0.5:
#                 if ref is not None:
#                     return img.transpose(Image.FLIP_LEFT_RIGHT), \
#                         mask.transpose(Image.FLIP_LEFT_RIGHT), \
#                         ref.transpose(Image.FLIP_LEFT_RIGHT),\
#                         edge.transpose(Image.FLIP_LEFT_RIGHT) 
#                 else:
#                     return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT),\
#                         edge.transpose(Image.FLIP_LEFT_RIGHT) 
#             if ref is not None:
#                 return img, mask, ref, edge 
#             return img, mask, edge 


# class Resize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)

#     def __call__(self, img, mask, manual_random=None, ref=None, edge=None):
#         assert img.size == mask.size
#         if ref is not None:
#             return img.resize(self.size, Image.BILINEAR), \
#                 mask.resize(self.size, Image.NEAREST), \
#                 ref.resize(self.size, Image.NEAREST), \
#                 edge.resize(self.size, Image.NEAREST)
#         return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), edge.resize(self.size, Image.NEAREST)




