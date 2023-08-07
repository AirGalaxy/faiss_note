# Fassi源码阅读
本节介绍IVFPQ方法
## 预备知识——什么是IVFPQ方法
前面我们已经介绍了IVF方法、PQ方法，那么将IVF、PQ方法结合在一起，就是IVFPQ方法。
### IVFPQ的入库过程
1. IVF过程  
设置聚类中心数量为nlist，我们会对所有的物料向量做聚类，按照上文所述的IVF方法得到nlist个倒排拉链，将物料向量分配到不同的倒排拉链中
2. 乘积量化(PQ)过程  
对所有向量做乘积量化，保存在倒排表中。  
当然Fassi对PQ过程是有优化的。在Fassi中，对于倒排拉链中的每一个向量，计算该向量与所属聚类中心的残差，得到残差向量。然后对**残差向量**做乘积量化。  
那么为什么要算残差，然后对残差做乘积量化呢? 我们借用一幅图来说明这个问题
<div align=center><img src="https://zhou-yuxin.github.io/articles/2020/IVFPQ%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/3.png"></div>  
上图中有10个聚类。如果先把原始的y物料向量变成残差，就会发现这些残差都是围绕在原点附近。该操作等效于把所有的聚类中心都平移到了原点，使得所有点都聚焦到原点。由于点变得更加聚拢，所以同样分为ksub个聚类时，平均而言每个聚类的区域会更小，做距离近似时，误差会更小。其实简单点说，如果聚类效果较好，残差的绝对大小比原始向量的绝对大小要小，对于绝对大小较小的数做量化，量化的误差肯定较小。  

此外还有个问题，我们做PQ时，是对聚类局部的残差向量分别做PQ，还是对所有的残差向量做PQ? 两种方案的唯一区别就在于PQ算法是聚类局部的（每个聚类都有自己的PQ）还是全局唯一的。第一种方案有一个明显的缺点，即训练时间会非常的长。以常用的IVF1024,PQ64为例，有1024个IVF的聚类，就要做1024次PQ算法。而每次PQ算法中，要对64个平面分别做聚类。因此需要执行非常多次的聚类算法，会消耗大量时间。事实上，Faiss也是采用了第二种方案。  
最终，在通过粗聚类形成了倒排拉链，倒排拉链中的id为全部物料的残差向量进行乘积量化后的id。  

### IVFPQ的搜索过程
很容易想到搜索时按照IVF和PQ进行两步搜索就行了:
1. 进行IVF搜索，找出需要进行检索的倒排拉链；
2. 计算query向量与聚类中心的残差，通过PQ搜索找到与query残差最近的k个物料残差，PQ搜索我们在PQ章节中已经分析过源码了。  

现在，我们可以来看看Fassi中IVFPQ的实现了

## IndexIVFPQ类
### 成员变量与构造函数
```c++
struct IndexIVFPQ : IndexIVF {
    //控制对原始物料向量量化还是对残差向量量化
    bool by_residual; 
    //乘积量化器
    ProductQuantizer pq; ///< produces the codes

    bool do_polysemous_training; ///< reorder PQ centroids after training?
    PolysemousTraining* polysemous_training; ///< if NULL, use default

    // search-time parameters
    size_t scan_table_threshold; ///< use table computation or on-the-fly?
    int polysemous_ht;           ///< Hamming thresh for polysemous filtering

    /** Precompute table that speed up query preprocessing at some
     * memory cost (used only for by_residual with L2 metric)
     */
    int use_precomputed_table;

    /// if use_precompute_table
    /// size nlist * pq.M * pq.ksub
    AlignedTable<float> precomputed_table;
}
```
