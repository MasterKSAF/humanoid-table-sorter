"""Revalidate a previously classified stationary tray against current depth."""
import numpy as np


def colored_rim_groups(points,hsv,plane):
    """Join disconnected walls across an occluding load, with current depth only."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points=np.asarray(points);plane=np.asarray(plane)
    height=points[:,2]-points[:,:2]@plane[:2]-plane[2]
    low=(height>.015)&(height<.09)&(hsv[:,2]>50)
    masks={'blue':(hsv[:,0]>90)&(hsv[:,0]<140)&(hsv[:,1]>40),
           'red':((hsv[:,0]<14)|(hsv[:,0]>170))&(hsv[:,1]>20),
           'green':(hsv[:,0]>45)&(hsv[:,0]<85)&(hsv[:,1]>40)}
    for color,mask in masks.items():
        p=points[low&mask]
        if len(p)<60:continue
        _,ids=np.unique(np.floor(p/.004).astype(int),axis=0,return_index=True);p=p[ids]
        pairs=cKDTree(p).query_pairs(.08,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(p),len(p)))
        count,labels=connected_components(graph,directed=False)
        for label in range(count):
            group=p[labels==label]
            if len(group)<30:continue
            h=group[:,2]-group[:,:2]@plane[:2]-plane[2]
            if np.quantile(h,.98)<.045:continue
            lo,hi=np.quantile(group,[.01,.99],axis=0)
            if not (.10<np.min(hi[:2]-lo[:2]) and np.max(hi[:2]-lo[:2])<.40):continue
            support=tray_support({'bounds':[lo,hi]},group)
            if support['confirmed']:yield color,group,lo,hi,support


def tray_support(item,points):
    bounds=np.asarray(item['bounds']);points=np.asarray(points)
    near=((points[:,:2]>=bounds[0,:2]-.012)&(points[:,:2]<=bounds[1,:2]+.012)).all(1)
    near &= (points[:,2]>bounds[1,2]-.025)&(points[:,2]<bounds[1,2]+.012)
    p=points[near];counts=[]
    for axis in (0,1):
        for edge in bounds[:,axis]:counts.append(int(np.count_nonzero(abs(p[:,axis]-edge)<.014)))
    # Three rectangle sides constrain its position. Two are ambiguous.
    return dict(confirmed=sum(n>=6 for n in counts)>=3,points_per_wall=counts)
