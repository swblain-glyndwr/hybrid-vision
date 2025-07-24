import torch, itertools, argparse
from torch.utils.data import DataLoader, TensorDataset
from src.common.codec import AdaptiveFlowEncoder, _ckpt

def main(dset_path, epochs=20, lr=1e-3, batch=128, mu=1e-3):
    feats = torch.load(dset_path)                # pre-extracted neck tensors
    loader = DataLoader(TensorDataset(feats), batch_size=batch, shuffle=True)
    model  = AdaptiveFlowEncoder(feats.shape[1])
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total, rec = 0., 0.
        for (x,) in loader:
            x = x.float()                     # ensure fp32
            if x.ndim == 1:                   
                x = x.unsqueeze(0)            # make it [1, D]
            loss, r = model.criterion(x, mu)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item(); rec += r
        print(f"epoch {epoch:02d} loss={total/len(loader):.4e} rec={rec/len(loader):.4e}")

    _ckpt.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), _ckpt)
    print(f"Saved trained AFE to {_ckpt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", required=True,
                   help="*.pt file of stacked YOLO neck tensors")
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()
    main(args.tensors, epochs=args.epochs)
