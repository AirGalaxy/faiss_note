# Fassi源码阅读
## 1. IndexIVFInterface
IndexIVFInterface是接口类，定义了使用IVF方法的index必须实现的方法。
```c++
struct IndexIVFInterface : Level1Quantizer {
    // 对于1个query检索nprobe个倒排拉链
    size_t nprobe = 1;    ///< number of probes at query time
    // 限制检索最多检索max_codes个物料向量
    size_t max_codes = 0; ///< max nb of codes to visit to do a query

    // 构造函数，和Level1Quantizer一致
    explicit IndexIVFInterface(Index* quantizer = nullptr, size_t nlist = 0)
        : Level1Quantizer(quantizer, nlist) {}
}
```
IndexIVFInterface是Level1Quantizer的子类，这样IndexIVFInterface也可以train()，得到物料向量的nlist个聚类中心(或者说倒排拉链)  

IndexIVFInterface定义了两个检索接口:  
```c++
virtual void search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params = nullptr,
        IndexIVFStats* stats = nullptr) const = 0;
```
|参数|含义|
|:---|:---:|
|n|有n个query检索|  
|x|query向量集合|  
|k|要检索离query最近的k个向量|  
|assign/key|要检索的聚类中心的编号，大小为 n * nprobe，类型为idx_t| 
|distances|检索到的物料向量离query向量的距离，大小为n * k，类型为float|
|centroid_dis|query向量到要检索的聚类中心的距离，大小为 n * nprobe，类型为float|
|labels|检索到的向量的唯一标识符(有的地方也叫id)，大小n * k|
|store_pairs|是否要存储倒排表的LO|
|params|进行IVF检索的参数|
|stats|输出参数，保存搜索的一些参数|

此外，还有一个range_search_preassigned()，该函数的参数含义和search_preassigned完全一致，只是多了一个radius参数，只要与query向量的距离在radius内的物料向量都被视作搜索结果  

可以想到，Fassi中把IVF检索分为了两步。第一步是获得最近的nprobe个聚类中心对应的倒排拉链，第二步是用query向量去这些倒排拉链中去进行检索，这一步也就是search_preassigned/range_search_preassigned函数要做的事

## 2. IndexIVF
### 2.1 简介
IndexIVF类是一个抽象类，提供了基本IVF检索功能的实现，是Index、IndexIVFInterface的子类。先看下其成员变量
```c++
    // 倒排表
    InvertedLists* invlists = nullptr;
    // 是否拥有该invlists
    bool own_invlists = false;
    // 每个向量占用多少个字节
    size_t code_size = 0; ///< code size per vector in bytes
    /** Parallel mode determines how queries are parallelized with OpenMP
     *
     * 0 (default): split over queries
     * 1: parallelize over inverted lists
     * 2: parallelize over both
     * 3: split over queries with a finer granularity
     *
     * PARALLEL_MODE_NO_HEAP_INIT: binary or with the previous to
     * prevent the heap to be initialized and finalized
     */
    int parallel_mode = 0;
    const int PARALLEL_MODE_NO_HEAP_INIT = 1024;

    // 在预备知识中已经说了，将物料向量的唯一标识符id映射为倒排表的LO
    DirectMap direct_map;
```
到这里成员变量大家应该很熟悉了，这里可能要解释下parallel_mode。  
在进行IVF检索时，Fassi提供了不同粒度的并行模式  
parallel_mode是一个控制并行化方式的参数。它决定了在执行查询过程中如何使用OpenMP进行并行化。

* 当parallel_mode为0时，默认值，查询过程将根据查询的数量进行划分，每个查询在一个线程中执行。这种方式在处理多个查询时比较高效。

* 当parallel_mode为1时，查询过程将并行化到倒排列表级别。也就是说，对于每个倒排列表，可以使用不同的线程进行处理，以加快查询的速度。这种方式在处理包含大量倒排列表的查询索引时比较有效。

* 当parallel_mode为2时，查询过程将同时应用上述两种方式的并行化策略，即同时在查询和倒排列表级别进行并行化。

* 当parallel_mode为3时，类似于模式0，但是在查询的划分上有更细的粒度，可以更充分地利用多个线程的计算资源，以进一步提高查询性能。

此外，还有一个可选的标志PARALLEL_MODE_NO_HEAP_INIT，可以与上述模式二进制"或"运算，以防止在并行化过程中初始化和销毁堆栈。这可以提高性能，但需要谨慎使用，以确保并行执行的正确性。

构造函数如下
```c++
IndexIVF::IndexIVF(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t code_size,
        MetricType metric)
        //向量维数d，采用的metric
        : Index(d, metric),
          // 使用的quantizer，有nlist个聚类中心/倒排拉链个数
          IndexIVFInterface(quantizer, nlist),
          // 使用ArrayInvertedLists实现的倒排表
          invlists(new ArrayInvertedLists(nlist, code_size)),
          // RAII，是否拥有invlists，决定在析构时，是否释放invlists
          own_invlists(true),
          code_size(code_size) {
    FAISS_THROW_IF_NOT(d == quantizer->d);
    // quantizer被训练，且quantizer中的所有存储的聚类中心向量的个数为nlist时才认为已被训练
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // 如果metric_type为内积，则在训练quantizer时，每次进行kmeans迭代训练时，进行normalize
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
}
```
directMap的type为NoMap且未分配空间

IndexIVF的工作流程:
1. 使用物料向量训练找到聚类中心
2. 添加物料向量
3. 搜索
接下来我会按找这三个步骤介绍IndexIVF是怎么实现的

### 2.2 训练
如预备知识中所述，我们需要一个Index类型的quantizer用来查询距离query向量最近的nprobe个聚类中心，并且需要对quantizer预先进行训练。  
```c++
void IndexIVF::train(idx_t n, const float* x) {
    train_q1(n, x, verbose, metric_type);

    train_residual(n, x);
    is_trained = true;
}
```
train_q1(n, x)计算物料向量的聚类中心。  
train_residual(n, x) 对物料向量到其所属的聚类中心的残差进行训练，在IndexIVF中不做任何事，在IndexIVFPQ中才会有效果。
注意IndexIVF是Level1Quantizer的子类，而train_q1我们已经在预备知识中分析过了，现在我们知道在IndexIVF::train()后quantizer中已经有了有了训练好的聚类中心。
### 2.3 添加物料向量
重载父类Index中的添加物料向量的方法:
```c++
void IndexIVF::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // RAII
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    // 找到最近的聚类中心点，放到coarse_idx中
    quantizer->assign(n, x, coarse_idx.get());
    // 把物料向量添加到最近聚类中心对应的倒排拉链中
    add_core(n, x, xids, coarse_idx.get());
}
```
xids为指定的物料向量的唯一标识符，传空指针时为自增的id  
 
所有的操作都需要通过add_core方法来进行。
```c++
void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx); 
```
n:添加n个物料向量
x:物料向量
xids:物料向量的id
coarse_idx:物料向量所属的聚类中心id

add_core的实现如下:
```c++
void IndexIVF::add_core( idx_t n, const float* x, const idx_t* xids,
        const idx_t* coarse_idx) {
   idx_t bs = 65536;
   //如果n的取值超过bs，则按照bs的大小分批处理
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            add_core(
                    i1 - i0,
                    x + i0 * d,
                    xids ? xids + i0 : nullptr,
                    coarse_idx + i0);
        }
        return;
    }

    // 将物料向量集合编码到flat_codes中
    std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
    encode_vectors(n, x, coarse_idx, flat_codes.get());

    DirectMapAdd dm_adder(direct_map, n, xids);
    #pragma omp parallel reduction(+ : nadd)
    {   // 总线程数
        int nt = omp_get_num_threads();
        // 当前线程编号 0，1，2，3...
        int rank = omp_get_thread_num();
        // nadd为实际添加到倒排表中的元素
        size_t nadd = 0, nminus1 = 0;
        for (size_t i = 0; i < n; i++) {
            //记录没有找到聚类中心的向量的个数，这些没找到聚类中心的向量不会加入倒排表中
             if (coarse_idx[i] < 0)
                 nminus1++;
         }
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            //根据物料向量对应的聚类中心的编号分配到不同的线程上去
            //这意味着一个线程操作一个倒排拉链，通过dm_adder保证对同一个倒排拉链操作的线程安全性
            if (list_no >= 0 && list_no % nt == rank) {
                // 从xids中取出id，否则使用自增id
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                        list_no, id, flat_codes.get() + i * code_size);
                // 同步更新Direct_map
                dm_adder.add(i, list_no, ofs);

                nadd++;
            // 如果list_no == -1，则不添加到倒排表中，只添加到DirectMap中
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }
    ntotal += n;
}
```
add_core的实现中，通过OpenMp的线程号将要处理的倒排拉链分散到不同线程而避免了竞争，这是一种无锁的并发模式，显著的提升了效率。  
这段代码中也有一些奇怪的地方，如对list_no/(coarse_idx[i]) < 0的情况的处理，理论上来说，这种情况会发生在IVFIndex没有train，或者trian的时候没有添加任何物料向量，就直接调用add_core，但是真正添加到倒排表中的元素nadd确只打了个log，计算ntotal时却是加上了所有的元素(包括list_no/(coarse_idx[i]) < 0的元素)个数n，这里也求大佬解惑！

### 2.4 搜索
首先分析最复杂的方法search_preassigned，这个函数的用处与参数的含义在IndexIVFInterface已经有分析，这里我们只看下其实现
```c++
void IndexIVF::search_preassigned(        
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) {
        //step1:根据IndexIVF本身的参数及 params 设置nprobe、max_codes、IDSelector
        ...

        //还记得上文说的parallel_mode取值0，1，2，3吗?在这种情况下，默认do_heap_init为true
        int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
        //只需将parallel_mode(取值范围0，1，2，3)左移10位就可以设置do_heap_init为false
        bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);
        //


    }
```



重载了Index类的方法:
```c++
void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
        const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
        }
```