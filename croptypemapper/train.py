from torch.autograd import Variable


def train(trainData, model, criterion, optimizer, gpu=True, train_loss=[]):

    model.train()
    epoch_loss = 0
    i = 0

    for s1_img, s2_img, label in trainData:
        s1_img = Variable(s1_img)
        s2_img = Variable(s2_img)
        label = Variable(label)

        if gpu:
            s1_img = s1_img.cuda()
            s2_img = s2_img.cuda()
            label = label.cuda()

        fused_model_out = model(s1_img, s2_img)
        fused_loss = criterion()(fused_model_out, label)
        epoch_loss += fused_loss.item()

        i += 1

        optimizer.zero_grad()
        fused_loss.backward()
        optimizer.step()

    print("train loss: {}".format(epoch_loss / i))
    if train_loss != None:
        train_loss.append(float(epoch_loss / i))
