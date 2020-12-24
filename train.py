
def train(*, model, ld, opt, criterion, epoch):
    for e in range(epoch):
        for i, (inputs, labels) in enumerate(ld):
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()