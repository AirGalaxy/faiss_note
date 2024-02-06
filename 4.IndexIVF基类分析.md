# Faiss源码阅读
本节开始分析IVF方法的实现
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

可以想到，Faiss中把IVF检索分为了两步。第一步是获得最近的nprobe个聚类中心对应的倒排拉链，第二步是用query向量去这些倒排拉链中去进行检索，这一步也就是search_preassigned/range_search_preassigned函数要做的事

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
在进行IVF检索时，Faiss提供了不同粒度的并行模式  
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
   //如果样本数量n超过bs，则按照bs的大小分批处理
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
        // 当前线程编号 0，1，2，3...， 通过list_no % nt == rank实现不同的倒排拉链分配到不同线程上
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
#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap) {
        // 每个并行线程获得一个InvertedListScanner
        InvertedListScanner* scanner =
        get_InvertedListScanner(store_pairs, sel);


        // step2. 初始化lambda，辅助接下来真正并行的操作
        // 这几个lambda中 simi为存放检索到的物料向量到query向量的距离， idxi为物料向量的id或LO
        // 初始化存放结果的堆的lambda
        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };
        // 添加结果到堆中的lambda
        auto add_local_results = [&](const float* local_dis,
                                 const idx_t* local_idx,
                                 float* simi,
                                 idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
            }
        }

        // 对结果进行堆排序的lambda
        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        // 下面这个lambda很重要，实现了扫描指定list_no的倒排拉链
        auto scan_one_list = [&](idx_t key, //指定倒排拉链的编号
                                 float coarse_dis_i,
                                 float* simi, //当前倒排拉链中的元素到query向量的距离
                                 idx_t* idxi, // 当前倒排拉链中
                                 idx_t list_size_max) {//每个倒排拉链最多检索的物料个数
            // 设置当前遍历的倒排拉链的编号
            scanner->set_list(key, coarse_dis_i);
            // 将扫描的倒排拉链的数量加1
            nlistv++;
            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;
                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key));
                    // 使用迭代器遍历倒排拉链，并在simi, idxi上根据物料向量到query向量的距离进行heapify
                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);
                } else {
                    //根据list_no获得倒排拉链的物料向量
                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();
                    // 如上所述，如果搜索结果中需要的是id，则从倒排拉链中取出物料id
                    if (!store_pairs) {
                        sids.reset(new InvertedLists::ScopedIds(invlists, key));
                        ids = sids->get();
                    }
                    // 如果是IDSelectorRange，则只检索IDSelectorRange范围内的物料向量
                    if (selr) {
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        // 放入到scanner堆中的元素个数，也就是scanner遍历的物料向量的个数
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }
                    // scan_codes遍历codes、ids，计算距离并保存最近的k个到simi、idxi这两个堆中，返回值为堆调整的次数
                    nheap += scanner->scan_codes(
                            list_size, codes, ids, simi, idxi, k);
                    //返回scanner遍历的物料向量的个数
                    return list_size;
                }
            }

        };

        //step.3 所有的lambda已经准备好，进行真正的并行操作
        // 根据pmode做不同的并行策略，这部分逻辑我们拆出来说
        ...
    }
    //并行段结束, 更新状态
    if (ivf_stats) {
        ivf_stats->nq += n;
        ivf_stats->nlist += nlistv;
        ivf_stats->ndis += ndis;
        ivf_stats->nheap_updates += nheap;
    }

}
```
pmode=0或3的情况，进行query粒度的并行:
```c++
#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap) {
        // 每个并行线程获得一个InvertedListScanner
        InvertedListScanner* scanner =
        get_InvertedListScanner(store_pairs, sel);

        if (pmode == 0 || pmode == 3) {
#pragma omp for
        // query粒度的并行
        for (idx_t i = 0; i < n; i++) {
            scanner->set_query(x + i * d);
            // 每次循环，指向存放结果的指针向前走一个query结果的大小(k)
            float* simi = distances + i * k;
            idx_t* idxi = labels + i * k;
            init_result(simi, idxi);
            idx_t nscan = 0;

            // 遍历倒排拉链时不并行(pmode=0 || pmode=3)
            for (size_t ik = 0; ik < nprobe; ik++) {
                // 遍历倒排拉链，将当前拉链中比结果堆中距离小的元素插入结果堆
                nscan += scan_one_list(
                        keys[i * nprobe + ik],
                        coarse_dis[i * nprobe + ik],
                        simi,
                        idxi,
                        max_codes - nscan);
                // 扫描的物料向量个数超过 max_codes，停止扫描
                if (nscan >= max_codes) {
                    break;
                }
            }

            ndis += nscan;
            // 对当前query对应的搜索结果的堆进行堆排序，这里是原地进行排序，需要o(1)的额外空间
            reorder_result(simi, idxi);
        }
    }
```

pmode=1，对倒排表进行并行的处理:
```c++
if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                // 每个线程只有一个scanner
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                
#pragma omp for schedule(dynamic)
                // 并行的for是以nprobe，也就是倒排表的粒度来并行的
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            //结果放到了thread_local的dis与idx数组中
                            local_dis.data(),
                            local_idx.data(),
                            unlimited_list_size);
                }
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                // 初始化最终结果的当前query对应的哪个堆。单线程执行
                init_result(simi, idxi);
// 每个线程都将等待，直到其他所有线程都达到此点。
#pragma omp barrier
// 每个线程按照单线程的顺序挨个把thread_local结果放到最终结果数组中
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier

// 只在一个线程中对当前query进行原址堆排序
#pragma omp single
                reorder_result(simi, idxi);
            }
        } 
```
当pmode = 2时，会进行query粒度及倒排拉链的粒度的并行处理:
```c++
if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            //在单线程上初始化最终结果数组
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
// 同时对倒排拉链与query进行并行的处理
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                //第几个倒排拉链
                size_t i = ij / nprobe;
                //第几个query向量
                size_t j = ij % nprobe;
                //设置scanner的query
                scanner->set_query(x + i * d);
                //初始化thread_local的结果
                init_result(local_dis.data(), local_idx.data());
                //计算距离并建堆
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size);
//各线程按顺序把thread_local的结果放到最终结果distances、labels中
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
//在单线程上对query结果进行原址排序
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } 
```
由于这里我们没有设置openMP的线程数，所以最大的并行的线程数还是会受到CPU核心数的限制。  
可以看到，这里的search_preassigned实现了对指定倒排拉链的搜索，同时使用openMP进行可控制粒度的并行操作，当然根据Faiss源码的注释中所写，最多的还是pmode=0，即query粒度的并行。

此外，还有range_search_preassigned方法，提供了范围检索的功能，其代码结构完全一致，只是在scan_codes时调用了scan_codes_range或iterate_codes_range方法，这里就不分析了，感兴趣的读者可以自己研究。


重载了Index类的方法:
```c++
void IndexIVF::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
        // 计算合适的nprobe
        const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
        // 定义sub_search_func这个lambda，这也是搜索的主要功能函数
        auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        // query要检索的倒排拉链编号
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        // query到这些聚类中心的距离
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);
        // 查找离query最近的nprobe个聚类中心的id(list_no)
        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);
        // 预取倒排表
        invlists->prefetch_lists(idx.get(), n * nprobe);
        // 如前文所述，根据上面quantizer->search()找到的聚类中心id去进行搜索
        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats);
    };
    // 如果按照query的粒度并行的话
    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        //按照openMP支持的最大线程数划分请求，分片进行搜索
        for (idx_t slice = 0; slice < nt; slice++) {
           sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
        }
    } else {
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}
```
如果你还记得本节第一部分最后的内容，那你就会发现search的逻辑和上面说的完全一致

最后简单介绍下range_search函数，这个函数我相信就算我不说大家也很清楚了，套路都是一样的:
```c++
void IndexIVF::range_search(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
            //step1.确定nprobe
            const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
            std::unique_ptr<idx_t[]> keys(new idx_t[nx * nprobe]);
            std::unique_ptr<float[]> coarse_dis(new float[nx * nprobe]);
            //step2.寻找聚类中心id
            quantizer->search(
                nx, x, nprobe, coarse_dis.get(), keys.get(), quantizer_params);
                    invlists->prefetch_lists(keys.get(), nx * nprobe);
            //step3.调用range_search_preassigned完成搜索
            range_search_preassigned(
                    nx,
                    x,
                    radius,
                    keys.get(),
                    coarse_dis.get(),
                    result,
                    false,
                    params,
                    &indexIVF_stats);
        }
```
range_search并没有分批搜索，实现是很朴素的。


### 2.5 辅助函数
```c++
void IndexIVF::add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids) {
    size_t coarse_size = coarse_code_size();
    DirectMapAdd dm_adder(direct_map, n, xids);

    for (idx_t i = 0; i < n; i++) {
        //被编码后的物料向量包括物料向量本身和聚类中心的编码值
        const uint8_t* code = codes + (code_size + coarse_size) * i;
        idx_t list_no = decode_listno(code);
        idx_t id = xids ? xids[i] : ntotal + i;
        size_t ofs = invlists->add_entry(list_no, id, code + coarse_size);
        dm_adder.add(i, list_no, ofs);
    }
    ntotal += n;
}
```
这个函数提供了添加物料向量的接口，可以看到IndexIVF默认被编码的向量是将物料向量和物料向量的聚类中心编码在一起的。

```c++
size_t IndexIVF::remove_ids(const IDSelector& sel) {
    //同时从倒排表和direct_map中删除在sel中的物料向量id
    size_t nremove = direct_map.remove_ids(sel, invlists);
    ntotal -= nremove;
    return nremove;
}
```
删除接口，通过DirectMap的remove_ids接口实现对倒排表和DirectMap的同步删除

```c++
void IndexIVF::update_vectors(int n, const idx_t* new_ids, const float* x) {
    // 如果DirectMap是hashtable，则先删除再添加
    if (direct_map.type == DirectMap::Hashtable) {
        IDSelectorArray sel(n, new_ids);
        size_t nremove = remove_ids(sel);
        add_with_ids(n, x, new_ids);
        return;
    }

    std::vector<idx_t> assign(n);
    //先找到物料对应的聚类中心
    quantizer->assign(n, x, assign.data());

    std::vector<uint8_t> flat_codes(n * code_size);
    //先编码
    encode_vectors(n, x, assign.data(), flat_codes.data());
    //直接更新倒排表和directmap
    direct_map.update_codes(
            invlists, n, new_ids, assign.data(), flat_codes.data());
}
```
update_vectors实现了对物料向量的更新，对于DirectMap类型的不同处理也不同

编码接口:
```c++
size_t IndexIVF::sa_code_size() const {
    size_t coarse_size = coarse_code_size();
    //IndexIVF中，被编码的物料向量长度为物料向量本身的长度+粗聚类中心的长度
    return code_size + coarse_size;
}

void IndexIVF::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    std::unique_ptr<int64_t[]> idx(new int64_t[n]);
    // 先找到粗聚类中心
    quantizer->assign(n, x, idx.get());
    // 然后才能编码
    encode_vectors(n, x, idx.get(), bytes, true);
}
```

重建向量的接口:
```c++
void IndexIVF::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params_in) const;
```
先搜索，然后重建搜索到的向量，其搜索部分和不使用并行化的搜索流程一致，最终调用``reconstruct_from_offset``实现重建，该接口IndexIVF并未实现，再IVFPQ和IndexIVFFlat中会介绍

```c++
void IndexIVF::copy_subset_to(
        IndexIVF& other,
        InvertedLists::subset_type_t subset_type,
        idx_t a1,
        idx_t a2) const {
            other.ntotal +=
            invlists->copy_subset_to(*other.invlists, subset_type, a1, a2);
        }
```
委托给invlists实现，已经在预备知识中说过，不再介绍

合并倒排表的接口:
```c++
void IndexIVF::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    IndexIVF* other = static_cast<IndexIVF*>(&otherIndex);
    invlists->merge_from(other->invlists, add_id);

    ntotal += other->ntotal;
    other->ntotal = 0;
}
```
委托给invlists实现，合并两个倒排表

## 3. 总结
1. IndexIVF作为一个抽象基类，实现了对IVF方法，并使用OpenMP在多线程上进行了优化，通过将不同的数据分配到不同的线程上，避免了竞争的出现。
2. IndexIVF将搜索与向量的存储、压缩实现分离开来，子类实现向量的压缩、存储即可实现IVF检索。