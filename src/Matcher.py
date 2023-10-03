import time
import faiss
import numpy as np

class Matcher():
    def __init__(self, k, vectors, logger):
        self.logger = logger
        self.vectors = np.squeeze(np.array(vectors))
        self.top_k = k
        self.dim = self.vectors.shape[-1]
        #print("matcher", self.vectors.shape)

    def Init1(self):
        #self.kmeans = faiss.Kmeans(self.dim, 50, niter=500, verbose=True, gpu=False)
        self.kmeans.train(self.vectors)
        self.cluster_cents = kmeans.centroids
        cluster_wucha = kmeans.obj
        print(cluster_cents) #各类中心点向量
        print(cluster_wucha)

        self.index = faiss.IndexFlatL2(self.dim)  # build the index
        #self.index = faiss.IndexFlatIP(self.dim)
        '''IndexFlatIP
        IndexHNSWFlat
        IndexIVFFlat
        IndexLSH
        IndexScalarQuantizer
        IndexPQ
        IndexIVFScalarQuantizer
        IndexIVFPQ
        IndexIVFPQR'''
        #print(self.index.is_trained)
        self.index.add(self.vectors)  # add vectors to the index
        #print(self.index.ntotal)

    def search1(self, input):
        D, I = self.index.search(input, self.top_k)  # sanity check
        #print(I)  # 向量索引位置
        #print(D)  # 相似度矩阵
        #print(D)
        return I, D

    def Init(self):
        self.index = faiss.IndexFlatL2(self.dim)  # build the index
        #self.index = faiss.IndexFlatIP(self.dim)
        '''IndexFlatIP
        IndexHNSWFlat
        IndexIVFFlat
        IndexLSH
        IndexScalarQuantizer
        IndexPQ
        IndexIVFScalarQuantizer
        IndexIVFPQ
        IndexIVFPQR'''
        #print(self.index.is_trained)
        self.index.add(self.vectors)  # add vectors to the index
        #print("faiss Init", self.index.ntotal)
        end = time.time()
        #print("use ", start-end)
        self.logger.debug("{} init success!".format(self.__class__.__name__))
        return 0

    def search(self, input):
        #print(self.top_k)
        #print(self.vectors.shape)
        D, I = self.index.search(input, self.top_k)  # sanity check
        #print(I)  # 向量索引位置
        #print(D)  # 相似度矩阵
        #print(D)
        return I[0], D[0]









