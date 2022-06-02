from torch.autograd import Variable

def validate(evalData, model, criterion, weights, gpu=True, val_loss=[]):
    model.eval()
    epoch_loss = 0
    i = 0
    # set_trace()
    for s1_img, s2_img, label in evalData:
        s1_img = Variable(s1_img, requires_grad=False)
        # s1_img[s1_img != s1_img] = -100
        s2_img = Variable(s2_img, requires_grad=False)
        # s2_img[s2_img != s2_img] = -100
        label = Variable(label, requires_grad=False)

        if gpu:
            s1_img = s1_img.cuda()
            s2_img = s2_img.cuda()
            label = label.cuda()

        # model_out = model(s1_img, s2_img)
        # loss = nn.CrossEntropyLoss()(model_out, label)
        # epoch_loss += loss.item()

        # s1_model_out,  s2_model_out= model(s1_img, s2_img)
        # s1_loss = criterion()(s1_model_out, label)
        # s2_loss = criterion()(s2_model_out, label)
        # s1_weight = 0.5
        # total_loss = s1_loss.item() * s1_weight + s2_loss.item() * (1 - s1_weight)
        # epoch_loss += total_loss

        s1_model_out, s2_model_out, fused_model_out = model(s1_img, s2_img)
        s1_loss = criterion(ignore_index=0)(s1_model_out, label)
        s2_loss = criterion(ignore_index=0)(s2_model_out, label)
        fused_loss = criterion(ignore_index=0)(fused_model_out, label)
        total_loss = (s1_loss * weights[0] + s2_loss * weights[1] + fused_loss * weights[2])
        epoch_loss += total_loss.item()

        # print("val: ", i, epoch_loss)
        i += 1

    print("validation loss: {}".format(epoch_loss / i))
    if val_loss != None:
        val_loss.append(float(epoch_loss / i))
