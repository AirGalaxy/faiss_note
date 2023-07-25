# Fassi源码阅读
## 1.采用IVF方法的Index
### 1.1 IVF方法简述
为了提高检索速度，我们可以牺牲一些准确度，只检索部分向量。我们希望选取离query向量近的一些向量。于是一个简单的想法是先对所有物料向量做聚类，得到N个聚类中心，在搜索时，先计算query向量与聚类中心的距离，找到最近的nlist个聚类中心，只与这nlist个聚类中心所代表的向量计算距离，找到其中最近的k个。这就是基于倒排表的向量检索方法。  
这里的倒排表只是借用了传统的全文检索的搜索引擎中的词，可以把聚类中心理解为lucene中的倒排拉链的term，把属于该聚类中心的向量理解为倒排拉链中的doc

### 1.2 辅助类介绍
#### 1.2.1 Level1Quantizer类
先考察Level1Quantizer类的成员变量
```c++
struct Level1Quantizer {
    /// quantizer that maps vectors to inverted lists
    // 把向量映射为倒排表的量化器
    Index* quantizer = nullptr;
    // 倒排表的数量
    size_t nlist = 0;

    /**
     * = 0: use the quantizer as index in a kmeans training
     * 等于0时，quantizer只是单纯的作为k-means聚类时的索引(调用quantizer的search接口找最近的向量)
     * = 1: just pass on the training set to the train() of the quantizer
     等于1时，把训练集(物料向量)传递给quantizer的train()来进行训练
     * = 2: kmeans training on a flat index + add the centroids to the quantizer
     等于2时，在上文的flatindex上做kmeans聚类，最后把聚类中心点加入到quantizer中
     */
    char quantizer_trains_alone = 0;

    //quantizer属于这个Level1Quantizer吗？属于的话在析构函数中释放
    bool own_fields = false;
    // 聚类时使用的index
    Index* clustering_index = nullptr;
}
```
这里解答原来的一个疑惑，为何quantizer是Index类型？难道Index也是一种量化器吗？  
如果我们把通过物料向量聚类得到的聚类中心向量加入到Index中，再在该Index上搜索离输入向量最近的向量，用搜索结果代替元素输入向量，这样我们就可以把输入向量量化到某个聚类中心上，从这个角度上看，把Index作为一种量化器是很自然而合理的事。

接下来考察其成员函数
```c++
    /// Trains the quantizer and calls train_residual to train sub-quantizers
    void train_q1(size_t n,const float* x,bool verbose,
            MetricType metric_type) {
        size_t d = quantizer->d;
        if (quantizer_trains_alone == 1) {
            //单纯训练quantizer，
            quantizer->train(n, x);
        } else if (quantizer_trains_alone == 0) {
            //新建一个Clustering，接下来介绍
            Clustering clus(d, nlist, cp);
            quantizer->reset();
            if (clustering_index) {
                //若clustering_index不为空，使用clus聚类
                clus.train(n, x, *clustering_index);
                //然后把训练结果得到的聚类中心加入到quantizer中
                quantizer->add(nlist, clus.centroids.data());
            } else {
                //在Clustering中介绍
                clus.train(n, x, *quantizer);
            }
            quantizer->is_trained = true;
        }  else if (quantizer_trains_alone == 2) {
            Clustering clus(d, nlist, cp);
            if (!clustering_index) {
                //clustering_index为空，使用IndexFlatL2
                IndexFlatL2 assigner(d);
                clus.train(n, x, assigner);
            } else {
                //否则使用clustering_index聚类
                clus.train(n, x, *clustering_index);
            }
            if (!quantizer->is_trained) {
                //使用上面训练出的聚类中心继续训练quantizer
                quantizer->train(nlist, clus.centroids.data());
            }
        //聚类中心加入quantizer
        quantizer->add(nlist, clus.centroids.data());
        }
    }
```
先看下train_q1的参数：  
n:有多少个要训练的向量  
x:向量数组  
metric_type:L2/内积/.etc  
代码结构很清晰，根据quantizer_trains_alone取值走不同的道路