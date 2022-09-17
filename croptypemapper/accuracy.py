import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
from .metrics import BinaryMetrics
from pathlib import Path

def accuracy_evaluation(eval_data, model, gpu, out_prefix, bucket=None):
    """
    Evaluate model
    Params:
        eval_data (''DataLoader'') -- Batch grouped data
        model -- Trained model for validation
        buffer: Buffer added to the targeted grid when creating dataset. This allows metrics to calculate only
            at non-buffered region
        gpu (binary,optional): Decide whether to use GPU, default is True
        bucket (str): name of s3 bucket to save metrics
        outPrefix (str): s3 prefix to save metrics
    """

    model.eval()
    metrics = []

    for s1_img, s2_img, label in eval_data:
        s1_img = Variable(s1_img, requires_grad=False)  # shape=(B,T,C)
        #s1_img[s1_img != s1_img] = 0
        s2_img = Variable(s2_img, requires_grad=False)
        #s2_img[s2_img != s2_img] = 0
        label = Variable(label, requires_grad=False)  # shape=1

        if gpu:
            s1_img = s1_img.cuda()
            s2_img = s2_img.cuda()
            label = label.cuda()

        # model_out = model(s1_img, s2_img) #shape=(B, Class_num)
        # model_out_prob = F.softmax(model_out, 1)
        # model_out_prob = F.softmax(out_logits, 1)

        out_logits = model(s1_img, s2_img)
        model_out_prob = F.softmax(out_logits, 1)

        batch, nclass = model_out_prob.size()

        for i in range(batch):
            label_batch = label[i].cpu().numpy()
            batch_pred = model_out_prob.max(dim=1)[1].data[i].cpu().numpy()

            for z in range(nclass):
                class_out = model_out_prob[:, z].data[i].cpu().numpy()
                class_pred = np.where(batch_pred == z, 1, 0)
                class_label = np.where(label_batch == z, 1, 0)
                pixel_metrics = BinaryMetrics(class_label, class_out, class_pred)

                try:
                    metrics[z].append(pixel_metrics)
                except:
                    metrics.append([pixel_metrics])
    # set_trace()
    metrics = [sum(m) for m in metrics]

    report = pd.DataFrame({
        "Overal Accuracy": [m.oa() for m in metrics],
        "Producer's Accuracy (recall)": [m.producers_accuracy() for m in metrics],
        "User's Accuracy (precision)": [m.users_accuracy() for m in metrics],
        "Negative Predictive Value": [m.npv() for m in metrics],
        "Specificity (TNR)": [m.specificity() for m in metrics],
        "F1 score": [m.f1_measure() for m in metrics],
        "IoU": [m.iou() for m in metrics],
        "mIoU": [m.miou() for m in metrics],
        "TSS": [m.tss() for m in metrics]
    }, index=["class_{}".format(m) for m in range(len(metrics))])

    if bucket:
        metrics_path = f"s3://{bucket}/{out_prefix}/Metrics.csv"
    else:
        metrics_path = Path(out_prefix).joinpath("Metrics.csv")
        Path(out_prefix).mkdir(parents=True, exist_ok=True)

    report.to_csv(metrics_path)
