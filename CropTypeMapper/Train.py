


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

        # model_out = model(s1_img, s2_img)
        # loss = criterion()(model_out, label)
        # epoch_loss += loss.item()

        # s1_model_out,  s2_model_out= model(s1_img, s2_img)
        # s1_loss = criterion()(s1_model_out, label)
        # s2_loss = criterion()(s2_model_out, label)
        # s1_weight = 0.5
        # total_loss = s1_loss * s1_weight + s2_loss * (1 - s1_weight)
        # epoch_loss += total_loss.item()

        s1_model_out, s2_model_out, fused_model_out = model(s1_img, s2_img)
        s1_loss = criterion()(s1_model_out, label)
        s2_loss = criterion()(s2_model_out, label)
        fused_loss = criterion()(fused_model_out, label)
        total_loss = (s1_loss * 0.35 + s2_loss * 0.45 + fused_loss * 0.2) / 3
        epoch_loss += total_loss.item()

        # print("train: ", i, epoch_loss)
        i += 1

        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        optimizer.step()

    print("train loss: {}".format(epoch_loss / i))
    if train_loss != None:
        train_loss.append(float(epoch_loss / i))
