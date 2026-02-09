import os
import pickle
import argparse
from sklearn import metrics
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
import numpy as np
import torch

def get_opt():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--file', default='all_ood_results.pkl', type=str)
    args = parser.parse_args()
    return args

def _cal_auc_fpr(id_ness, labels):
    auroc = metrics.roc_auc_score(labels, id_ness)
    fpr,tpr,thresh = Roc(labels, id_ness, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr

to_np = lambda x: x.detach().cpu().numpy()
def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0])
def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0])
def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1))


def evaluate_ood_results(ood_results):
    """评估 OOD 任务输出的结果"""
    # ground-truth
    label_key_name = "label"
    labels = ood_results[label_key_name]  # [n, 1]

    eval_results = {}
    for key, value in ood_results.items():
        if key == label_key_name:
            continue
        auroc, fpr = _cal_auc_fpr(id_ness=value, labels=labels)
        eval_results[key] = (auroc, fpr)
    
    return eval_results



def eval_ood(data):
    """
    data: 模型输出的预测结果，该文件是一个 dict，具有如下两个字段：
              - "label": 形状为 [N] 的 numpy 数组, 表示 ground-truth, 1 表示 ID, 0 表示 OOD
              - "logit": 形状为 [N, 117] 的 numpy 数组, 表示模型输出的动作类别置信度分数
              - "atd": 形状为 [N, 1] 的 numpy 数组, 可选
              - "ctw": 形状为 [N, 1] 的 numpy 数组, 可选
    """
    label = data["label"]
    logit = torch.from_numpy(data["logit"])

    # ID 样本和 OOD 样本的比例
    pos_cnt = np.count_nonzero(label)
    neg_cnt = label.shape[0] - pos_cnt
    print(f"ID/OOD: {pos_cnt}/{neg_cnt}")

    ind_logits = max_logit_score(logit)
    ind_prob = msp_score(logit)
    ind_energy = energy_score(logit)

    ood_results = {
        "label": label,
        "MSP": ind_prob,
        "MaxLogit": ind_logits,
        "Energy": ind_energy
    }

    if "atd" in data.keys():
        ood_results["ATD"] = data["atd"]
    if "ctw" in data.keys():
        ood_results["CTW"] = data["ctw"]

    ood_performance = evaluate_ood_results(ood_results)
    for metric_name, (auroc, fpr) in ood_performance.items():
        print(f"{metric_name}: auroc={auroc*100:.2f}, fpr={fpr*100:.2f}")
    return ood_performance

if __name__ == "__main__":
    args = get_opt()
    with open(args.file, "rb") as f:
        data = pickle.load(f)
    eval_ood(data)
