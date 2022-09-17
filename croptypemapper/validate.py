from torch.autograd import Variable

def validate(evalData, model, criterion, gpu=True, val_loss=[]):
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

        fused_model_out = model(s1_img, s2_img)
        fused_loss = criterion()(fused_model_out, label)
        epoch_loss += fused_loss.item()

        # print("val: ", i, epoch_loss)
        i += 1

    print("validation loss: {}".format(epoch_loss / i))
    if val_loss != None:
        val_loss.append(float(epoch_loss / i))
