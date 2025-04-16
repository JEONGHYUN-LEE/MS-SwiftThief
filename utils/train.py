import torch
from tqdm import tqdm


def samples_per_class(loader, model, num_classes, device):
    model.eval()
    cnt = torch.zeros(num_classes)
    for x in loader:
        x = x[0]
        x = x.to(device).float()
        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        cnt += torch.bincount(predicts, minlength=num_classes).detach().cpu()
    return cnt


def train(loader, model, optimizer, criterion, device):
    model.train()
    corrects = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        optimizer.zero_grad()
        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return float(corrects) / float(len(loader.dataset))


def train_kd(loader, model, teacher, optimizer, device):
    model.train()
    teacher.eval()
    corrects = 0
    for x in tqdm(loader):
        x = x[0]
        x = x.to(device).float()

        with torch.no_grad():
            lesson = teacher(x)
        output = model(x)
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(output, dim=1),
            torch.nn.functional.softmax(lesson, dim=1),
            reduction='batchmean'
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict = output.argmax(dim=1)
        t_predict = lesson.argmax(dim=1)

        corrects += (predict == t_predict).sum().item()

    return float(corrects) / float(len(loader.dataset))


def train_kd_hard(loader, model, teacher, optimizer, device):
    model.train()
    teacher.eval()
    corrects = 0
    for x in tqdm(loader):
        x = x[0]
        x = x.to(device).float()

        with torch.no_grad():
            lesson = teacher(x)
        output = model(x)
        # loss = torch.nn.functional.kl_div(
        #     torch.nn.functional.log_softmax(output, dim=1),
        #     torch.nn.functional.softmax(lesson, dim=1),
        #     reduction='batchmean'
        # )
        criterion = torch.nn.CrossEntropyLoss().to(device)
        loss = criterion(output, lesson.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict = output.argmax(dim=1)
        t_predict = lesson.argmax(dim=1)

        corrects += (predict == t_predict).sum().item()

    return float(corrects) / float(len(loader.dataset))


def test(loader, model, device):
    model.eval()
    corrects = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        outputs = model(x)
        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

    return float(corrects) / float(len(loader.dataset))
