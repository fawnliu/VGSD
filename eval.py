import os, argparse
from numpy import *
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from glob import glob

def listdirs_only(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    
class Metrics:
    def __init__(self):
        self.initial()
    def initial(self):
        self.tp = []
        self.tn = []
        self.fp = []
        self.fn = []
        self.precision = []
        self.recall = []
        self.cnt = 0
        self.mae = []
        self.tot = []
    def update(self, pred, target, name):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        assert pred.all()>=0.0 and pred.all()<=1.0
        assert target.all()>=0.0 and target.all()<=1.0
        assert pred.shape==target.shape
        if any(target) is False:
            print(name)
            return # do not calculate empty GTs
        
        ## threshold = 0.5 
        TP = lambda prediction, true: sum(logical_and(prediction, true))
        TN = lambda prediction, true: sum(logical_and( logical_not(prediction), logical_not(true) ) )
        FP = lambda prediction, true: sum(logical_and(logical_not(true), prediction))
        FN = lambda prediction, true: sum(logical_and(logical_not(prediction), true))
        
        trueThres = 0.5
        predThres = 0.5
        self.tp.append( TP(pred>=predThres, target>trueThres) )
        self.tn.append( TN(pred>=predThres, target>trueThres) )
        self.fp.append( FP(pred>=predThres, target>trueThres) )
        self.fn.append( FN(pred>=predThres, target>trueThres) )
        self.tot.append( target.shape[0] )
        assert self.tot[-1]==(self.tp[-1]+self.tn[-1]+self.fn[-1]+self.fp[-1])
        
        if self.tp[-1] + self.fp[-1] +self.fn[-1] == 0:
            print(name)
        ## 256 precision and recall
        tmp_prec = []
        tmp_recall = []
        eps = 1e-9

        trueHard = target>0.5

        bins = linspace(0, 255, 256)
        fg_hist, _ = histogram(pred[trueHard], bins=bins)  # 最后一个bin为[255, 256]
        bg_hist, _ = histogram(pred[~trueHard], bins=bins)

        fg_w_thrs = cumsum(flip(fg_hist), axis=0)
        bg_w_thrs = cumsum(flip(bg_hist), axis=0)

        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs

        # 为防止除0，这里针对除0的情况分析后直接对于0分母设为1，因为此时分子必为0
        # Ps[Ps == 0] = 1
        T = max(count_nonzero(target), 1)

        # TODO: T=0 或者 特定阈值下fg_w_thrs=0或者bg_w_thrs=0，这些都会包含在TPs[i]=0的情况中，
        #  但是这里使用TPs不便于处理列表
        precisions = (TPs + eps) / (Ps + eps)
        recalls = (TPs + eps) / (T + eps)


        # for threshold in range(256):
        #     threshold = threshold / 255.
        #     tp = TP(pred>=threshold, trueHard)+eps
        #     ppositive = sum(pred>=threshold)+eps
        #     tpositive = sum(trueHard)+eps
        #     tmp_prec.append( tp/ppositive )
        #     tmp_recall.append( tp/tpositive )

        self.precision.append(precisions)
        self.recall.append(recalls)

        ## mae
        self.mae.append( mean(abs(pred-target)) )
        
        self.cnt += 1
    
    def compute_iou(self):
        iou = []
        n = len(self.tp)
        for i in range(n):
            iou.append(self.tp[i]/(self.tp[i]+self.fp[i]+self.fn[i]))
        return mean(iou)
    def compute_fbeta(self, beta_square=0.3):
        precision = array(self.precision).mean(axis=0)
        recall = array(self.recall).mean(axis=0)
        max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])
        return max_fmeasure
    def compute_mae(self):
        return mean(self.mae)
    def accuracy(self):
        return array([(self.tp[i]+self.tn[i])/self.tot[i] for i in range(len(self.tot))]).mean()
    def ber(self):
        return array([100*(1.0-0.5*( self.tp[i]/(self.tp[i]+self.fn[i]) + self.tn[i]/(self.tn[i]+self.fp[i]) )) for i in range(len(self.tot))]).mean()
    def report(self):
        # report = "Count:"+str(self.cnt)+"\n"
        report = "IOU:{}, f1:{}, MAE:{}, accuracy:{}, BER:{}\n".format(self.compute_iou(),\
                                                                    self.compute_fbeta(), \
                                                                   self.compute_mae(),\
                                                                   self.accuracy(),\
                                                                   self.ber() )
        return report



def func(gt_name, name):
    # global gt_img_name, pred_img_name
    met = Metrics()
    # gt_name = gt_img_name[idx]
    # name = pred_img_name[idx]
    gt = array(Image.open(gt_name).convert('L')) 
    pred = array(Image.open(name).convert('L'))
    # .astype(uint8)
    # print("gt", gt.max())
    
    gt_max = 255 if gt.max()>127. else 1.0
    gt = gt / gt_max
    # pred_max = pred.max()
    # if pred.max() == 255:
    #     print("pred", pred.max())

    ####
    pred = pred.astype(float) / 255.
    ####

    eps = 1e-9
    # if pred_max == 0.0:
    #     pred = pred.astype(float) / (pred_max + eps)
    # else:
    #     pred = pred.astype(float) / pred_max
    
    met.update(pred=pred, target=gt, name=name)
    return met

parser = argparse.ArgumentParser()
parser.add_argument("-pred", "--prediction", type=str, default=None)  #results/
parser.add_argument("-exp", "--exp", type=str, default="VMD_ours")
parser.add_argument("-gt", "--mirrormask", type=str, default=None)
parser.add_argument("-db", "--dataset", type=str, default="VMD")
args = parser.parse_args()
db_name = args.dataset

print(args.mirrormask, '=====')
print(args.prediction)

pred_path = args.prediction
merge_metrics = Metrics()
pred_img_name_ref = glob(os.path.join(pred_path, "*", "*.png"))
pred_img_name = []
## filter reflection out
for item in pred_img_name_ref:
    if '_ref.png' in item:
        continue
    pred_img_name.append(item)

gt_img_name = glob(os.path.join(args.mirrormask, "*", "SegmentationClassPNG", "*.png"))
if len(pred_img_name) != len(gt_img_name):
    print("pred", len(pred_img_name), "gt", len(gt_img_name))
    raise ValueError("pred and gt not match")

pred_img_name = sorted(pred_img_name)
gt_img_name = sorted(gt_img_name)

n = len(pred_img_name)
num_worker = 16
with Parallel(n_jobs=num_worker) as parallel:
    metric_lst = parallel( delayed(func)(gt_name, pred_name) 
                          for gt_name, pred_name in tqdm(zip(gt_img_name, pred_img_name), total=n) )
for x in metric_lst:
    merge_metrics.tp += x.tp
    merge_metrics.tn += x.tn
    merge_metrics.fp += x.fp
    merge_metrics.fn += x.fn
    merge_metrics.precision += x.precision
    merge_metrics.recall += x.recall
    merge_metrics.cnt += x.cnt
    merge_metrics.mae += x.mae
    merge_metrics.tot += x.tot
# for video in sorted(os.listdir(pred_path)):
#     pred_img_name = [x for x in os.listdir(os.path.join(pred_path, video)) if x.endswith(".png")]
#     gt_img_name = [x for x in os.listdir(os.path.join(args.mirrormask, video,  "SegmentationClassPNG")) if x.endswith(".png")]
#     n = len(pred_img_name)
#     num_worker = 16
#     with Parallel(n_jobs=num_worker, prefer="threads") as parallel:
#         metric_lst = parallel( delayed(func)(i, pred_path, video) for i in tqdm(range(n)))
#     for x in metric_lst:
#         merge_metrics.tp += x.tp
#         merge_metrics.tn += x.tn
#         merge_metrics.fp += x.fp
#         merge_metrics.fn += x.fn
#         merge_metrics.precision += x.precision
#         merge_metrics.recall += x.recall
#         merge_metrics.cnt += x.cnt
#         merge_metrics.mae += x.mae
#         merge_metrics.tot += x.tot

log = merge_metrics.report()
print(log)
# os.makedirs(args.exp, exist_ok=True)
# with open(os.path.join(args.exp, "fast_report.txt"), "a") as f:
#     f.write(pred_path + " " + log)
