# Fassi源码阅读
本节继续介绍PQ方法的源码
## 预备知识——对称检索和非对称检索
假设我们在进行乘积量化的过程中，向量被分割为M列子向量，每列子向量有ksub个聚类中心。  
物料向量被乘积量化后，只存储了聚类中心的索引，考虑一个query向量。当我们要检索离query向量最近的k个物料向量时，我们会想到两种策略:
1. 一种朴素的方法是，把乘积量化的后物料向量索引(ProductQuantizer::compute_code得到的结果)先还原为物料向量(在Index类中被称为reconstruct)，然后暴力搜索所有的物料向量，找到最近的k个  
如果我们使用的L2距离，我们可以想到一些用内存换速度的优化方法。在L2距离下，两个向量的L2距离的平方等于其子向量L2距离的平方之和。因此，我们可以先算子向量的L2距离的平方再加在一起。  
由于所有子向量的可能取值为M * ksub，对于一个query向量，我们将其分割为M个子向量，然后计算M个子向量到对应位置上的所有ksub个聚类中心的距离，最后我们得到一个M * ksub大小的距离表dis_tables，接下来对于物料向量，我们只需要查询dis_tables就可以查到在同一列的query子向量与物料向量的子向量的距离。
## 3. ProductQuantizer类
接下来介绍PQ类的最后一类方法:检索方法
```c++
    /** perform a search (L2 distance)
     * @param x        query vectors, size nx * d
     * @param nx       nb of queries
     * @param codes    database codes, size ncodes * code_size
     * @param ncodes   nb of nb vectors
     * @param res      heap array to store results (nh == nx)
     * @param init_finalize_heap  initialize heap (input) and sort (output)?
     */
    void search(
            const float* x,
            size_t nx,
            const uint8_t* codes,
            const size_t ncodes,
            float_maxheap_array_t* res,
            bool init_finalize_heap = true) const {
        std::unique_ptr<float[]> dis_tables(new float[nx * ksub * M]);
        // 计算出query向量的子向量到子向量聚类中心的距离
        compute_distance_tables(nx, x, dis_tables.get());
        // knn检索
        pq_knn_search_with_tables<CMax<float, int64_t>>(
            *this,
            nbits,
            dis_tables.get(),
            codes,
            ncodes,
            res,
            init_finalize_heap);
    }
```
照例先看下参数:
|参数|含义|
|:---:|:---:|
|x|query向量数组|
|nx|query向量个数|
|codes|物料向量编码后的数组|
|ncodes|物料向量个数|
|res|存放结果的大顶堆|

代码结构很简单，先计算距离表再调用pq_knn_search_with_tables去检索最近的物料向量，上文说过，compute_distance_tables计算的是子向量到子聚类中心的距离，接下来看看pq_knn_search_with_tables的实现
```c++
template <class C>
void pq_knn_search_with_tables(
        const ProductQuantizer& pq,
        size_t nbits,
        const float* dis_tables,
        const uint8_t* codes,
        const size_t ncodes,
        HeapArray<C>* res,
        bool init_finalize_heap) {
    //k为每个堆分配的大小
    //nx为query向量的个数，即res中有多少个堆？
    size_t k = res->k, nx = res->nh;
    //ksub子向量维数，M为子向量列数
    size_t ksub = pq.ksub, M = pq.M;
// 按照query向量的粒度并行
#pragma omp parallel for if (nx > 1)
    for (int64_t i = 0; i < nx; i++) {
        //按照compute_distance_tables，把指针移动到当前query向量对应的子向量距离表的位置
        const float* dis_table = dis_tables + i * ksub * M;

        //指针移动到存放结果的堆的位置
        int64_t* __restrict heap_ids = res->ids + i * k;
        float* __restrict heap_dis = res->val + i * k;
        //有需要的话，初始化堆
        if (init_finalize_heap) {
            heap_heapify<C>(k, heap_dis, heap_ids);
        }
        pq_estimators_from_tables_generic<C>(
                pq,
                nbits,
                codes,
                ncodes,
                dis_table,
                k,
                heap_dis,
                heap_ids);
        
        if (init_finalize_heap) {
            heap_reorder<C>(k, heap_dis, heap_ids);
        }

    }
 }
```
