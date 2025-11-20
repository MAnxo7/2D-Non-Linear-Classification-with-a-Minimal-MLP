import torch,argparse,os
import sklearn.datasets as skd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from src import utils,models,train
from src import viz
# Entry point: dataset creation, model setup, training, visualization.
#ARGS 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--device", type=str, default=utils.get_device())
parser.add_argument("--ckpt-path", type=str, default=None)
parser.add_argument("--weight-decay", type=float, default=0)
args = parser.parse_args()

#BASIC CONFIG
utils.set_seed(0,deterministic=True)

epochs = args.epochs
batch = args.batch_size
lr = args.lr
device = args.device
weight_decay = args.weight_decay

# DATA PREPARATION
circles_train_x,circles_train_y = skd.make_circles(n_samples=500,noise=0.05) 

circles_eval_x,circles_eval_y = skd.make_circles(n_samples=125,noise=0.05) 

# DATALOADERS CREATION
# First position pattern
X_train = torch.tensor(circles_train_x,dtype=torch.float32).to(device)
Y_train = torch.tensor(circles_train_y,dtype=torch.float32).to(device)

X_eval = torch.tensor(circles_eval_x,dtype=torch.float32).to(device)
Y_eval = torch.tensor(circles_eval_y,dtype=torch.float32).to(device)

Y_train,Y_eval = Y_train.reshape([Y_train.shape[0],1]),Y_eval.reshape([Y_eval.shape[0],1]),

dataset_train = TensorDataset(X_train,Y_train)
dataset_eval = TensorDataset(X_eval,Y_eval)


dataloader_train = DataLoader(dataset_train,shuffle=True,batch_size=batch)
dataloader_eval = DataLoader(dataset_eval,shuffle=False,batch_size=batch)

#SPECS
model = models.BasicNN(16,X_train.shape[1],1,device)

opt = torch.optim.Adam(params=model.parameters(),lr=lr,weight_decay=weight_decay)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

#TRAIN OR EVAL
if args.eval_only:
    if args.ckpt_path is None:
        raise ValueError("You should specific --ckpt-path when use --eval-only")
    utils.load_checkpoint(args.ckpt_path,model,opt)
    val_metrics = train.evaluate(model,dataloader_eval,loss_fn,device)
    
else:
    run_dir = train.fit(model,device,dataloader_train,dataloader_eval,opt,loss_fn,epochs,early_stopping=20)
    viz.visualize_fig(model,circles_eval_x,circles_eval_y,device,os.path.join(run_dir,"figures","boundary.jpg"))



