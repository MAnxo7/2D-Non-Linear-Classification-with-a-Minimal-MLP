import matplotlib.pyplot as plt
import csv
import os, torch
# Visualization utilities: loss/acc curves and 2D decision boundaries.
def plot_from_csv(csv_path):
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    savepath = os.path.split(csv_path)[0]
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'train':
                train_loss.append(float(row['loss']))
                train_acc.append(float(row['acc']))
            elif row['split'] == 'eval':
                val_loss.append(float(row['loss']))
                val_acc.append(float(row['acc']))
                epochs.append(int(row['epoch']))
    visualize_loss(train_loss, val_loss, save_path=os.path.join(savepath,'figures','loss.jpg'))
    visualize_acc(train_acc, val_acc, save_path=os.path.join(savepath,'figures','acc.jpg'))
    
def visualize_loss(train_loss, val_loss, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(train_loss, label="Train Loss", linewidth=2)
    plt.plot(val_loss, label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(save_path):
        plt.savefig(save_path)
    plt.show()


def visualize_acc(train_acc, val_acc, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.plot(train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(val_acc, label="Validation Accuracy", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if(save_path):
        plt.savefig(save_path)
    plt.show()
        
    
def visualize_fig(model,eval_x,eval_y,device, save_path=None):
    pad = 0.2
    min_x = min(eval_x[:,0])-pad
    min_y = min(eval_x[:,1])-pad
    
    max_x = max(eval_x[:,0])+pad
    max_y = max(eval_x[:,1])+pad

    linspace_x = torch.linspace(min_x,max_x,300)
    linspace_y = torch.linspace(min_y,max_y,300)

    xx,yy = torch.meshgrid([linspace_x,linspace_y],indexing='ij')

    grid = torch.stack([xx.ravel(),yy.ravel()],dim=-1)
    grid = grid.to(device)
    
    
    model_was_training = model.training
    
    model.eval()
    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits)
        
    Z = probs.reshape(shape=xx.shape)
    Z = Z.cpu().numpy()

    xx_np, yy_np = xx.cpu().numpy(), yy.cpu().numpy()
    
    if(model_was_training):
        model.train()
        
    plt.contourf(xx_np, yy_np, Z, levels=25, cmap="coolwarm", alpha=0.6)
    plt.contour(xx_np, yy_np, Z, levels=[0.5], colors="black", linewidths=2)

    # Class 0
    plt.scatter(
        eval_x[eval_y == 0, 0],   # X Class 0
        eval_x[eval_y == 0, 1],   # Y Class 0
        c="blue",
        s=10,
        label="Class 0"
    )

    # Class 1
    plt.scatter(
        eval_x[eval_y == 1, 0],   # X Class 1
        eval_x[eval_y == 1, 1],   # Y Class 1
        c="yellow",
        s=10,
        label="Class 1"
    )

    plt.legend()
    if(save_path):
        plt.savefig(save_path)
    plt.show()

