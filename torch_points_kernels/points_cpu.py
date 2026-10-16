import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix

def ball_query(support, query, radius, max_num=-1, mode=0):
    # support: [N, 3] or [B, N, 3]
    # query: [M, 3] or [B, M, 3]
    is_batched = (support.dim() == 3)
    if is_batched:
        B = support.shape[0]
        inds = []
        dists = []
        for b in range(B):
            ind_b, dist_b = ball_query(support[b], query[b], radius, max_num, mode)
            inds.append(ind_b.unsqueeze(0))
            dists.append(dist_b.unsqueeze(0))
        return torch.cat(inds, dim=0), torch.cat(dists, dim=0)

    device = support.device
    support_np = support.detach().cpu().numpy()
    query_np = query.detach().cpu().numpy()
    
    tree = KDTree(support_np)
    
    if mode == 0:
        neighbors = tree.query_ball_point(query_np, r=radius)
        M = query_np.shape[0]
        
        if max_num <= 0:
            max_num = max(len(n) for n in neighbors) if neighbors else 0
            if max_num == 0:
                max_num = 1
                
        ind = np.zeros((M, max_num), dtype=np.int64)
        dist = -np.ones((M, max_num), dtype=np.float32)
        
        for i, neighbors_i in enumerate(neighbors):
            if len(neighbors_i) > 0:
                n_idx = np.array(neighbors_i)
                diff = support_np[n_idx] - query_np[i]
                d2 = np.sum(diff**2, axis=-1)
                
                sort_idx = np.argsort(d2)
                n_idx = n_idx[sort_idx]
                d2 = d2[sort_idx]
                
                n_keep = min(len(n_idx), max_num)
                ind[i, :n_keep] = n_idx[:n_keep]
                dist[i, :n_keep] = d2[:n_keep]
                
                if n_keep < max_num:
                    ind[i, n_keep:] = n_idx[0]
                    
        return torch.from_numpy(ind).to(device), torch.from_numpy(dist).to(device)
        
    elif mode == 1:
        neighbors = tree.query_ball_point(query_np, r=radius)
        
        ind_pairs = []
        dist_vals = []
        
        for j, neighbors_j in enumerate(neighbors):
            for i in neighbors_j:
                ind_pairs.append([i, j])
                diff = support_np[i] - query_np[j]
                dist_vals.append([np.sum(diff**2)])
                
        if len(ind_pairs) == 0:
            ind = torch.zeros((0, 2), dtype=torch.long, device=device)
            dist = torch.zeros((0, 1), dtype=torch.float, device=device)
            return ind, dist
            
        ind = torch.tensor(ind_pairs, dtype=torch.long, device=device)
        dist = torch.tensor(dist_vals, dtype=torch.float, device=device)
        return ind, dist

def dense_knn(support, query, k=1):
    is_batched = (support.dim() == 3)
    if is_batched:
        B = support.shape[0]
        inds = []
        dists = []
        for b in range(B):
            ind_b, dist_b = dense_knn(support[b], query[b], k)
            inds.append(ind_b.unsqueeze(0))
            dists.append(dist_b.unsqueeze(0))
        return torch.cat(inds, dim=0), torch.cat(dists, dim=0)

    device = support.device
    support_np = support.detach().cpu().numpy()
    query_np = query.detach().cpu().numpy()
    
    tree = KDTree(support_np)
    dists, inds = tree.query(query_np, k=k)
    
    if k == 1:
        dists = dists[:, np.newaxis]
        inds = inds[:, np.newaxis]
        
    return torch.from_numpy(inds).to(device), torch.from_numpy(dists**2).to(device)

def region_grow(xyz, labels, batch, ignore_labels=None, radius=0.04, nsample=200, min_cluster_size=10):
    device = xyz.device
    xyz_np = xyz.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    batch_np = batch.detach().cpu().numpy()
    
    if ignore_labels is not None:
        if torch.is_tensor(ignore_labels):
            ignore_labels = ignore_labels.detach().cpu().numpy()
        else:
            ignore_labels = np.array(ignore_labels)
    else:
        ignore_labels = np.array([])
        
    unique_batches = np.unique(batch_np)
    all_clusters = []
    
    for b in unique_batches:
        batch_mask = (batch_np == b)
        valid_mask = batch_mask
        for ignore_lbl in ignore_labels:
            valid_mask = valid_mask & (labels_np != ignore_lbl)
            
        if not np.any(valid_mask):
            continue
            
        valid_indices = np.where(valid_mask)[0]
        xyz_batch = xyz_np[valid_mask]
        labels_batch = labels_np[valid_mask]
        
        unique_labels = np.unique(labels_batch)
        for lbl in unique_labels:
            lbl_mask = (labels_batch == lbl)
            lbl_indices = valid_indices[lbl_mask]
            
            if len(lbl_indices) < min_cluster_size:
                continue
                
            xyz_lbl = xyz_batch[lbl_mask]
            tree = KDTree(xyz_lbl)
            pairs = tree.query_pairs(r=radius, output_type='ndarray')
            
            num_points = len(xyz_lbl)
            if len(pairs) == 0:
                continue
                
            row = np.concatenate([pairs[:, 0], pairs[:, 1]])
            col = np.concatenate([pairs[:, 1], pairs[:, 0]])
            data = np.ones(len(row), dtype=bool)
            
            adj = csr_matrix((data, (row, col)), shape=(num_points, num_points))
            n_components, comp_labels = csgraph.connected_components(
                csgraph=adj, directed=False, connection='weak'
            )
            
            for c in range(n_components):
                comp_mask = (comp_labels == c)
                comp_indices = lbl_indices[comp_mask]
                if len(comp_indices) >= min_cluster_size:
                    all_clusters.append(torch.from_numpy(comp_indices).long().to(device))
                    
    return all_clusters

def instance_iou(clusters, instance_labels, batch):
    device = instance_labels.device
    num_clusters = len(clusters)
    max_gt = int(torch.max(instance_labels).item())
    if max_gt <= 0:
        return torch.zeros((num_clusters, 0), device=device)
        
    iou_matrix = torch.zeros((num_clusters, max_gt), device=device)
    for i, cluster in enumerate(clusters):
        gt_labels_in_cluster = instance_labels[cluster]
        unique_gts, counts = torch.unique(gt_labels_in_cluster, return_counts=True)
        for gt, count in zip(unique_gts, counts):
            gt_val = int(gt.item())
            if gt_val <= 0:
                continue
            gt_mask = (instance_labels == gt_val)
            gt_indices = torch.where(gt_mask)[0]
            intersection = count.item()
            union = len(cluster) + len(gt_indices) - intersection
            if union > 0:
                iou_matrix[i, gt_val - 1] = intersection / union
    return iou_matrix
