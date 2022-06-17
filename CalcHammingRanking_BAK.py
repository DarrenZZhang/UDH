import numpy as np
import torch
import pdb

def CalcHammingDist_cuda(B1, B2, valid_bit_mask=None):
    if valid_bit_mask is not None:
        q = (valid_bit_mask != 0).sum()
        B1 = B1 * valid_bit_mask
    else:
        q = B2.shape[1]
    distH = 0.5 * (q - torch.mm(B1, B2.t()).squeeze())
    return distH

def CalcMap_cuda(qB, rB, queryL, retrievalL, valid_bit_mask=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (torch.mm(queryL[None, iter, :], retrievalL.t()) > 0).squeeze()
        tsum = torch.sum(gnd).cpu().data.numpy()
        if tsum == 0:
            continue
        if valid_bit_mask is None:
            hamm = CalcHammingDist_cuda(qB[None, iter, :], rB)
        else: 
            hamm = CalcHammingDist_cuda(qB[None, iter, :], rB, valid_bit_mask=valid_bit_mask[None, iter, :])
        ind = np.argsort(hamm.cpu().data.numpy())
        gnd = gnd[ind]
        gnd = gnd.cpu().data.numpy()
        # print(tsum)
        tsum = int(tsum)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map
    
def CalcMap_cuda_R(qB, rB, queryL, retrievalL, R=2, valid_bit_mask=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (torch.mm(queryL[None, iter, :], retrievalL.t()) > 0).squeeze()
        if valid_bit_mask is None:
            hamm = CalcHammingDist_cuda(qB[None, iter, :], rB)
        else: 
            hamm = CalcHammingDist_cuda(qB[None, iter, :], rB, valid_bit_mask=valid_bit_mask[None, iter, :])
        gnd = gnd[hamm <= R]
        hamm = hamm[hamm <= R]
        tsum = torch.sum(gnd).cpu().data.numpy()
        if tsum == 0:
            continue
        ind = np.argsort(hamm.cpu().data.numpy())
        gnd = gnd[ind]
        gnd = gnd.cpu().data.numpy()
        # print(tsum)
        tsum = int(tsum)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map    
    
    
def CalcL2Dist_cuda(B1, B2, valid_bit_mask=None):
    distH = (B1 - B2).pow(2)
    if valid_bit_mask is not None:
        distH = distH * valid_bit_mask
    distH = distH.sum(-1)
    return distH    
    
def CalcMap_cuda_R_rerank(qB, rB, qC, rC, queryL, retrievalL, R=2, valid_bit_mask=None):
    # qB: {-1,+1}^{mxq}, B for binary code
    # rB: {-1,+1}^{nxq}
    # qC: {-1,+1}^{mxq}, C for continuous code
    # rC: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (torch.mm(queryL[None, iter, :], retrievalL.t()) > 0).squeeze()
        if valid_bit_mask is None:
            hamm = CalcHammingDist_cuda(qB[None, iter, :], rB)
        else: 
            hamm = CalcHammingDist_cuda(qB[None, iter, :], rB, valid_bit_mask=valid_bit_mask[None, iter, :])
        gnd = gnd[hamm <= R]
        tsum = torch.sum(gnd).cpu().data.numpy()
        if tsum == 0:
            continue
        if valid_bit_mask is None:
            l2m = CalcL2Dist_cuda(qC[None, iter, :], rC[hamm <= R, :])
        else: 
            l2m = CalcL2Dist_cuda(qC[None, iter, :], rC[hamm <= R, :], valid_bit_mask=valid_bit_mask[None, iter, :])
        ind = np.argsort(l2m.cpu().data.numpy())
        gnd = gnd[ind]
        gnd = gnd.cpu().data.numpy()
        # print(tsum)
        tsum = int(tsum)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map       

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcMap(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        # print(tsum)
        tsum = int(tsum)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

def CalcReAcc(qB, rB,topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{mxq}
    # for test set
    num_query = qB.shape[0]
    topkacc = 0
    count = 0
    for index in range(num_query):
        hamm = CalcHammingDist(qB[index, :], rB)
        ind = np.argsort(hamm)
        tind = ind[0:topk]
        #index_s = index // 5
        index_s = index 
        count += index_s in tind
       
    topkacc = count / (num_query*1.0)

    return topkacc

def CalcTopAcc(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkacc = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        
        gnd = gnd[ind]
        
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        topkacc += tsum / topk
    topkacc = topkacc / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkacc


    

if __name__=='__main__':
    qB = np.array([[1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1]])
    rB = rB = np.array([
        [ 1,-1,-1,-1],
        [-1, 1, 1,-1],
        [ 1, 1, 1,-1],
        [-1,-1, 1, 1],
        [ 1, 1,-1,-1],
        [ 1, 1, 1,-1],
        [-1, 1,-1,-1]])
    queryL = np.array([
        [1,0,0],
        [1,1,0],
        [0,0,1],
    ], dtype=np.int64)
    retrievalL = np.array([
        [0,1,0],
        [1,1,0],
        [1,0,1],
        [0,0,1],
        [0,1,0],
        [0,0,1],
        [1,1,0],
    ], dtype=np.int64)

    topk = 5
    # map = CalcMap(qB, rB, queryL, retrievalL)
    # topkmap = CalcTopMap(qB, rB, queryL, retrievalL, topk)
    # print(map)
    # print(topkmap)
    acc =CalcTopAcc(qB, rB, queryL, retrievalL, topk)
    print(acc)
