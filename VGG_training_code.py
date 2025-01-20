import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# because torch-summary does not compatible with current pytorch
# use the successor of torch-summary -> torchinfo
import torchinfo
from torchinfo import summary


train_tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

test_tfm = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_tfm
)
val_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_tfm
)

device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size*2, shuffle=False, num_workers=2)

model_1 = models.vgg19_bn(num_classes=10).to(device=device)
summary(model_1)


from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device=None):
    total_time = end - start
    print(f"Train time on {device}: {total_time}")


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

train_epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),lr=0.0001)

from tqdm.auto import tqdm

train_losses = []
val_losses = []
train_accs = []
val_accs = []

train_time_start = timer()

torch.manual_seed(42)
torch.cuda.manual_seed(42)
for epoch in tqdm(range(train_epochs)):
    print(f"Epoch: {epoch} \n ")
    train_loss = 0
    train_acc = 0
    for batch, (X_train, y_train) in enumerate(train_dataloader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        model_1.train()
        train_pred = model_1(X_train)
        t_loss = loss_fn(train_pred, y_train)
        train_loss += t_loss.cpu().detach().numpy()
        train_acc += accuracy_fn(y_true=y_train, y_pred=train_pred.argmax(dim=1))
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    val_loss, val_acc = 0, 0
    model_1.eval()
    with torch.inference_mode():
        for X_test, y_test in val_dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            val_pred = model_1(X_test)
            v_loss = loss_fn(val_pred, y_test)
            val_loss += v_loss.cpu().detach().numpy()
            val_acc += accuracy_fn(y_true=y_test, y_pred=val_pred.argmax(dim=1))

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)
    print(f"\nTrain loss: {train_loss:.4f},  val loss: {val_loss:.4f},train acc: {train_acc:.4f}, val acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

train_time_end = timer()

total_train_time_model_0 = print_train_time(start=train_time_start,
                                                end=train_time_end,
                                                device=str(next(model_1.parameters()).device))



torch.save(model_1.state_dict(), "CIFAR10_model.pth")


num_epoch = [x for x in range(train_epochs)]

plt.plot(num_epoch, train_losses, label="train")
plt.plot(num_epoch, val_losses, label="val")
plt.title("Loss")
plt.legend()
plt.show()

plt.plot(num_epoch, train_accs, label="train")
plt.plot(num_epoch, val_accs, label="val")
plt.title("Accuracy")
plt.legend()
plt.show()