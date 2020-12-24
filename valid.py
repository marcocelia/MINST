import torch
import torch.nn.functional as F

# esegue un passo di test
def test(tls, net, test_ld):

    net.eval()
    test_loss = 0
    correct = 0

    # evita che si calcolino i gradienti
    with torch.no_grad():
        # visto che i dati di test sono 10.000 e il batch_size_tset e'
        # 1000 la valutazione e' divisa in 10 passi
        for data, target in test_ld:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_ld.dataset)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_ld.dataset),
        100. * correct / len(test_ld.dataset)))
    return test_loss