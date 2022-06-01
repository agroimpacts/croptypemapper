

def inference(testData, model, score_path, pred_path, gpu, test_label):
    # set_trace()

    testData, tile_id = testData

    meta_hard = {'driver': 'GTiff',
                 'dtype': 'uint8',
                 'nodata': None,
                 'width': 64,
                 'height': 64,
                 'count': 1,
                 'crs': None,
                 'transform': rasterio.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                 }

    meta_soft = meta_hard.copy()
    meta_soft.update({
        "dtype": "float32"
    })

    name_prob = "prob_{}.tif".format(tile_id)
    name_crisp = "crisp_{}.tif".format(tile_id)

    model.eval()

    h_canvas = np.zeros((1, meta_hard['height'], meta_hard['width']), dtype=meta_hard["dtype"])
    canvas_softScore_ls = []
    metrics = []

    if test_label:

        for s1_img, s2_img, coor, label in testData:
            s1_img = Variable(s1_img, requires_grad=False)
            s2_img = Variable(s2_img, requires_grad=False)
            label = Variable(label, requires_grad=False)

            if gpu:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                label = label.cuda()

            s1_model_out, s2_model_out, fused_model_out = model(s1_img, s2_img)
            pred_logits = (s1_model_out * 0.35 + s2_model_out * 0.45 + fused_model_out * 0.2) / 3
            pred_prob = F.softmax(pred_logits, 1)

            batch, nclass = pred_prob.size()

            for i in range(batch):
                index = (int(coor[0][i]), int(coor[1][i]))
                batch_label = label[i].cpu().numpy()
                out_predict = pred_prob.max(dim=1)[1].cpu().numpy()
                out_predict = np.expand_dims(out_predict, axis=0).astype(np.int8)
                h_canvas[:, index[0], index[1]] = out_predict

                for n in range(1, nclass):
                    out_softScore = pred_prob[:, n].data[i].cpu().numpy() * 100
                    out_softScore = np.expand_dims(out_softScore, axis=0).astype(np.float32)
                    class_label = np.where(batch_label == n, 1, 0)
                    chip_metrics = BinaryMetrics(class_label, out_softScore)

                    try:
                        metrics[n - 1].append(chip_metrics)
                    except:
                        metrics.append([chip_metrics])

                    try:
                        canvas_softScore_ls[n][:, index[0], index[1]] = out_softScore
                    except:
                        canvas_softScore_single = np.zeros((1, meta_soft['height'], meta_soft['width']),
                                                           dtype=meta_soft["dtype"])
                        canvas_softScore_single[:, index[0], index[1]] = out_softScore
                        canvas_softScore_ls.append(canvas_softScore_single)

        metrics = [sum(m) for m in metrics]
        filename = "Inference_metric_{}.csv".format(tile_id)

        report = pd.DataFrame({
            "Overal Accuracy": [m.oa() for m in metrics],
            "Producer's Accuracy (recall)": [m.producers_accuracy() for m in metrics],
            "User's Accuracy (precision)": [m.users_accuracy() for m in metrics],
            "Negative Predictive Value": [m.npv() for m in metrics],
            "Specificity (TNR)": [m.specificity() for m in metrics],
            "F1 score": [m.F1_measure() for m in metrics],
            "IoU": [m.iou() for m in metrics],
            "TSS": [m.tss() for m in metrics]},
            index=["class_{}".format(m) for m in range(1, len(metrics) + 1)])

        report.to_csv(filename, index=False)

    else:

        for s1_img, s2_img, coor in testData:
            s1_img = Variable(s1_img, requires_grad=False)
            s2_img = Variable(s2_img, requires_grad=False)

            if gpu:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()

            s1_model_out, s2_model_out, fused_model_out = model(s1_img, s2_img)
            pred_logits = (s1_model_out * 0.35 + s2_model_out * 0.45 + fused_model_out * 0.2) / 3
            pred_prob = F.softmax(pred_logits, 1)

            batch, nclass = pred_prob.size()

            for i in range(batch):
                index = (int(coor[0][i]), int(coor[1][i]))
                out_predict = pred_prob.max(dim=1)[1].cpu().numpy()
                out_predict = np.expand_dims(out_predict, axis=0).astype(np.int8)
                h_canvas[:, index[0], index[1]] = out_predict

                for n in range(nclass - 1):
                    out_softScore = pred_prob[:, n + 1].data[i].cpu().numpy() * 100
                    out_softScore = np.expand_dims(out_softScore, axis=0).astype(np.float32)
                    try:
                        canvas_softScore_ls[n][:, index[0], index[1]] = out_softScore
                    except:
                        canvas_softScore_single = np.zeros((1, meta_soft['height'], meta_soft['width']),
                                                           dtype=meta_soft["dtype"])
                        canvas_softScore_single[:, index[0], index[1]] = out_softScore
                        canvas_softScore_ls.append(canvas_softScore_single)

    with rasterio.open(Path(pred_path) / name_crisp, "w", **meta_hard) as dst:
        dst.write(h_canvas)

    for n in range(1, len(canvas_softScore_ls)):
        with rasterio.open(Path(score_path) / name_prob, "w", **meta_soft) as dst:
            dst.write(canvas_softScore_ls[n])
