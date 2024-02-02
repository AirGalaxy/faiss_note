# Faiss源码阅读
本节介绍IVFPQ方法
## 预备知识——什么是IVFPQ方法
前面我们已经介绍了IVF方法、PQ方法，那么将IVF、PQ方法结合在一起，就是IVFPQ方法。
### IVFPQ的入库过程
1. IVF过程  
设置聚类中心数量为nlist，我们会对所有的物料向量做聚类，按照上文所述的IVF方法得到nlist个倒排拉链，将物料向量分配到不同的倒排拉链中
2. 乘积量化(PQ)过程  
对所有向量做乘积量化，保存在倒排表中。  
当然Faiss对PQ过程是有优化的。在Faiss中，对于倒排拉链中的每一个向量，计算该向量与所属聚类中心的残差，得到残差向量。然后对**残差向量**做乘积量化。  
那么为什么要算残差，然后对残差做乘积量化呢? 我们借用一幅图来说明这个问题
<div align=center><img src="https://zhou-yuxin.github.io/articles/2020/IVFPQ%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/3.png"></div>  
上图中有10个聚类。如果先把原始的y物料向量变成残差，就会发现这些残差都是围绕在原点附近。该操作等效于把所有的聚类中心都平移到了原点，使得所有点都聚焦到原点。由于点变得更加聚拢，所以同样分为ksub个聚类时，平均而言每个聚类的区域会更小，做距离近似时，误差会更小。其实简单点说，如果聚类效果较好，残差的绝对大小比原始向量的绝对大小要小，对于绝对大小较小的数做量化，量化的误差肯定较小。  

此外还有个问题，我们做PQ时，是对聚类局部的残差向量分别做PQ，还是对所有的残差向量做PQ? 两种方案的唯一区别就在于PQ算法是聚类局部的（每个聚类都有自己的PQ）还是全局唯一的。第一种方案有一个明显的缺点，即训练时间会非常的长。以常用的IVF1024,PQ64为例，有1024个IVF的聚类，就要做1024次PQ算法。而每次PQ算法中，要对64个平面分别做聚类。因此需要执行非常多次的聚类算法，会消耗大量时间。事实上，Faiss也是采用了第二种方案。  
最终，在通过粗聚类形成了倒排拉链，倒排拉链中的id为全部物料的残差向量进行乘积量化后的id。  

### IVFPQ的搜索过程
很容易想到搜索时按照IVF和PQ进行两步搜索就行了:
1. 进行IVF搜索，找出需要进行检索的倒排拉链；
2. 计算query向量与聚类中心的残差，通过PQ搜索找到与query残差最近的k个物料残差，PQ搜索我们在PQ章节中已经分析过源码了。  

现在，我们可以来看看Faiss中IVFPQ的实现了

## IndexIVFPQ类
### 成员变量与构造函数
```c++
struct IndexIVFPQ : IndexIVF {
    //控制对原始物料向量量化还是对残差向量量化
    bool by_residual; 
    //乘积量化器
    ProductQuantizer pq; 
    // 参考IndexPQ，使用汉明距离对物料向量进行过滤
    bool do_polysemous_training; 
    PolysemousTraining* polysemous_training; 

    // search-time parameters
    size_t scan_table_threshold; ///< use table computation or on-the-fly?
    // 汉明门限
    int polysemous_ht; 

    /**
     * 是否要使用预计算的距离表?只有再L2范数距离的情况下才有效，这个不解释
     */
    int use_precomputed_table;

    /**
    * 内存对齐的查找表，存储预计算的聚类向量之间的距离
    * 使用的还是我们熟悉的AlignedTable，字节对齐的表
    * 大小为nlist * pq.M * pq.ksub
    * nlist:聚类后倒排拉链的个数   
    * pq.M:PQ后一个向量被切成M个子向量
    * pq.ksub:pq后子向量的维数
    */
    AlignedTable<float> precomputed_table;

```

### 索引接口 

添加向量:

先看下父类的接口

``` c++
// xids物料的标识符
// n本次要添加的物料向量的个数
void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    //找到物料向量属于哪一个聚类中心
    quantizer->assign(n, x, coarse_idx.get());
    //子类的add_core实现添加逻辑
    add_core(n, x, xids, coarse_idx.get());
}
```

add_core在子类IndexIVFPQ中的实现如下:

```c++
void IndexIVFPQ::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx) {
    add_core_o(n, x, xids, nullptr, coarse_idx);
}

void IndexIVFPQ::add_core_o(
        idx_t n,
        const float* x,
        const idx_t* xids,
        float* residuals_2,
        const idx_t* precomputed_idx) {
    idx_t bs = index_ivfpq_add_core_o_bs;
    //每批次处理index_ivfpq_add_core_o_bs(默认32768)个物料向量,代码略过
    ...
        
    direct_map.check_can_add(xids);

    const idx_t* idx;
    std::unique_ptr<idx_t[]> del_idx;
    if (precomputed_idx) {
        idx = precomputed_idx;
    // 如果还没计算好向量属于那个聚类中心，在这里再算一下
    // idx为该向量所属的聚类中心/倒排拉链
    } else {
        idx_t* idx0 = new idx_t[n];
        del_idx.reset(idx0);
        quantizer->assign(n, x, idx0);
        idx = idx0;
    }
    
    // to_encode为存放要被PQ编码结果的地址
    const float* to_encode = nullptr;
    std::unique_ptr<const float[]> del_to_encode;

    if (by_residual) {
        //如果要对残差进行PQ，在这里计算残差
        //计算残差就是把聚类中心编号idx先重建回原始的向量然后与物料向量x相减
        del_to_encode = compute_residuals(quantizer, n, x, idx);
        to_encode = del_to_encode.get();
    } else {
        to_encode = x;
    }
    //xcodes为存放编码结果的对象
    std::unique_ptr<uint8_t[]> xcodes(new uint8_t[n * code_size]);
    //将残差向量转换为PQ编码的向量放到xcodes中
    pq.compute_codes(to_encode, xcodes.get(), n);
    
    size_t n_ignore = 0;
    for (size_t i = 0; i < n; i++) {
        //聚类中心编号key
        idx_t key = idx[i];
        //物料向量唯一编号id,若外部没有输入，则使用自增id
        idx_t id = xids ? xids[i] : ntotal + i;
        // 没找到聚类中心
        if (key < 0) {
            //只添加到direct_map不添加到倒排表
            direct_map.add_single_id(id, -1, 0);
            //忽略的物料向量+1
            n_ignore++;
            if (residuals_2)
                //2级残差的值置为0
                memset(residuals_2, 0, sizeof(*residuals_2) * d);
            continue;
        }
		// 取IVFPQ编码后的结果
        uint8_t* code = xcodes.get() + i * code_size;
        // key:倒排拉链中心编号; id:唯一标识符; code:PQ编码
        size_t offset = invlists->add_entry(key, id, code);

        if (residuals_2) {
            
            float* res2 = residuals_2 + i * d;
            const float* xi = to_encode + i * d;
            pq.decode(code, res2);
            for (int j = 0; j < d; j++)
                res2[j] = xi[j] - res2[j];
        }
        direct_map.add_single_id(id, key, offset);
    }
}
```
PS:上面代码对key<0的情况的判断没搞懂, 什么时候会出现这种情况, key<0意味着在搜索k个最近邻向量时, 只找到了小于k个最近邻向量, 没找到的位置其label就是-1。但是对于IVFPQ方法来说，理论上物料向量一定属于一个聚类中心

### 训练接口

直接看父类IndexIVF的对train的实现:

```c++
void IndexIVF::train(idx_t n, const float* x) {
    if (verbose)
        printf("Training level-1 quantizer\n");
	// 第一次量化训练(找聚类中心)
    train_q1(n, x, verbose, metric_type);

    train_residual(n, x);
    is_trained = true;
}
```

代码很简单，在IndexIVFPQ中，``train_residual(n, x)``由``train_residual_o(idx_t n, const float* x, float* residuals_2)``实现:

```c++
void IndexIVFPQ::train_residual_o(idx_t n, const float* x, float* residuals_2) {
    const float* x_in = x;
    //可以看到这里如果用于训练的物料向量很多的话会对输入的物料向量进行了采样
    x = fvecs_maybe_subsample(
            d,
        	//如果采样了的话，n也随着变化
            (size_t*)&n,
        	//每个聚类中心最多对应多少个物料向量 * 子量化器的个数
            pq.cp.max_points_per_centroid * pq.ksub,
            x,
            verbose,
            pq.cp.seed);
        const float* trainset;
    
    ScopeDeleter<float> del_residuals;
    if (by_residual) {
		//进行残差的训练
        //先计算物料向量对应的粗聚类中心
        idx_t* assign = new idx_t[n]; // assignement to coarse centroids
        ScopeDeleter<idx_t> del(assign);
        //物料向量对应的聚类中心的编号放到assign中
        quantizer->assign(n, x, assign);
        float* residuals = new float[n * d];
        del_residuals.set(residuals);
        for (idx_t i = 0; i < n; i++)
            //计算残差
            quantizer->compute_residual(
                    x + i * d, residuals + i * d, assign[i]);
		//最终的训练集为残差
        trainset = residuals;
    } else {
        trainset = x;
    }
    //计算残差的PQ编码, 参考PQ部分的分析
    pq.train(n, trainset);
    //参考IndexIVF中，计算物料向量的汉明编码，搜索时通过汉明编码的编辑距离进行粗筛
    if (do_polysemous_training) {
    	PolysemousTraining default_pt;
    	PolysemousTraining* pt = polysemous_training;
    	if (!pt)
        	pt = &default_pt;
    	pt->optimize_pq_for_hamming(pq, n, trainset);
    }
}
```