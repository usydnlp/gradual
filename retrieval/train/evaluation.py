import torch
import numpy as np
from tqdm import tqdm
from timeit import default_timer as dt

from ..utils import layers


@torch.no_grad()
def predict_loader(model, data_loader, device):
    img_embs, cap_embs, q2_embs, v2_embs, cap_lens = None, None, None, None, None
    max_n_word = 77
    model.eval()

    pbar_fn = lambda x: x
    if model.master:
        pbar_fn = lambda x: tqdm(
            x, total=len(x),
            desc='Pred  ',
            leave=False,
        )

    for batch in pbar_fn(data_loader):
        ids = batch['index']
        if len(batch['caption'][0]) == 2:
            (_, _), (_, lengths) = batch['caption']
        else:
            cap, lengths = batch['caption']
        #img_emb, cap_emb, sg_embed =
        q1_emb, q2_emb, v1_emb, v2_emb = model.forward_batch(batch)

        if img_embs is None:
            if len(v1_emb.shape) == 3:
                is_tensor = True
                img_embs = np.zeros((len(data_loader.dataset), v1_emb.size(1), v1_emb.size(2)))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, q1_emb.size(2)))
                if q2_emb is not None:
                    q2_embs = np.zeros((len(data_loader.dataset), q2_emb.size(-1)))
                    q2_embs[ids, :] = q2_emb.data.cpu().numpy()
                if v2_emb is not None:
                    v2_embs = np.zeros((len(data_loader.dataset), v2_emb.size(-1)))
                    v2_embs[ids, :] = v2_emb.data.cpu().numpy()
            else:
                is_tensor = False
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = v1_emb.data.cpu().numpy()


        if is_tensor:
            cap_embs[ids,:max(lengths),:] = q1_emb.data.cpu().numpy()

        else:
            cap_embs[ids,] = q1_emb.data.cpu().numpy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]

    if img_embs.shape[0] == cap_embs.shape[0]:
        img_embs = remove_img_feat_redundancy(img_embs, data_loader)
        if v2_emb is not None:
            v2_embs = remove_img_feat_redundancy(v2_embs, data_loader)

    return cap_embs, q2_embs, img_embs, v2_embs, cap_lens

def remove_img_feat_redundancy(img_embs, data_loader):
        return img_embs[np.arange(
                            start=0,
                            stop=img_embs.shape[0],
                            step=data_loader.dataset.captions_per_image,
                        ).astype(np.int)]

@torch.no_grad()
def evaluate(
    model, q1_emb, q2_emb, v1_emb, v2_emb, lengths,
    device, shared_size=128, return_sims=False
):
    model.eval()
    _metrics_ = ('r1', 'r5', 'r10', 'medr', 'meanr')

    begin_pred = dt()

    v1_emb = torch.FloatTensor(v1_emb).to(device)
    q1_emb = torch.FloatTensor(q1_emb).to(device)
    if q2_emb is not None:
        q2_emb = torch.FloatTensor(q2_emb).to(device)
    if v2_emb is not None:
        v2_emb = torch.FloatTensor(v2_emb).to(device)

    end_pred = dt()
    sims = model.get_sim_matrix_shared(
        q1_emb=q1_emb, q2_emb=q2_emb, v1_emb=v1_emb, v2_emb=v2_emb,
        lens=lengths, shared_size=shared_size
    )
    sims = layers.tensor_to_numpy(sims)
    end_sim = dt()

    i2t_metrics = i2t(sims)
    t2i_metrics = t2i(sims)
    rsum = np.sum(i2t_metrics[:3]) + np.sum(t2i_metrics[:3])

    i2t_metrics = {f'i2t_{k}': v for k, v in zip(_metrics_, i2t_metrics)}
    t2i_metrics = {f't2i_{k}': v for k, v in zip(_metrics_, t2i_metrics)}
    print("## i2t_metrics")
    print(i2t_metrics)
    print("## t2i_metrics")
    print(t2i_metrics)
    print("## rsum")
    print(rsum)

    metrics = {
        'pred_time': end_pred-begin_pred,
        'sim_time': end_sim-end_pred,
    }
    metrics.update(i2t_metrics)
    metrics.update(t2i_metrics)
    metrics['rsum'] = rsum

    if return_sims:
        return metrics, sims

    return metrics


def i2t(sims):
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts
    print("## ------- [i2t]")
    print("Sims shape")
    print(sims.shape)
    print("captions_per_image: "+str(captions_per_image))

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        begin = captions_per_image * index
        end = captions_per_image * index + captions_per_image
        for i in range(begin, end, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr)


def t2i(sims):
    npts, ncaps = sims.shape
    captions_per_image = ncaps // npts

    ranks = np.zeros(captions_per_image * npts)
    top1 = np.zeros(captions_per_image * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    for index in range(npts):
        for i in range(captions_per_image):
            inds = np.argsort(sims[captions_per_image * index + i])[::-1]
            ranks[captions_per_image * index + i] = np.where(inds == index)[0][0]
            top1[captions_per_image * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr)
