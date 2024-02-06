# Faiss源码阅读
本节继续介绍PQ方法的源码
## 预备知识——对称检索和非对称检索
假设我们在进行乘积量化的过程中，向量被分割为M列子向量，每列子向量有ksub个聚类中心。  
物料向量被乘积量化后，只存储了聚类中心的索引，考虑一个query向量。当我们要检索离query向量最近的k个物料向量时，我们会想到两种策略:
1. 一种朴素的方法是，把乘积量化的后物料向量索引(ProductQuantizer::compute_code得到的结果)先还原为物料向量(在Index类中被称为reconstruct)，然后搜索所有的物料向量，找到最近的k个。这种方法被称作**非对称检索**   
如果我们使用的L2距离，我们可以想到一些用内存换速度的优化方法。在L2距离下，两个向量的L2距离的平方等于其子向量L2距离的平方之和。因此，我们可以先算子向量的L2距离的平方再加在一起。  
由于所有子向量的可能取值为M * ksub，对于一个query向量，我们将其分割为M个子向量，然后计算M个子向量到对应位置上的所有ksub个聚类中心的距离，最后我们得到一个M * ksub大小的距离表dis_tables，接下来对于物料向量，我们只需要查询dis_tables就可以查到在同一列的query子向量与物料向量的子向量的距离，然后将子向量的距离平方加起来，就得到了query向量与物料向量的L2距离的平方。 

2. 另一种方法是，我们将query向量也进行乘积量化，我们计算query向量量化后的向量与物料向量的距离，这种方法被称作**对称检索**。  
由于量化后的query向量的每个子向量的取值只能在对应列子向量的ksub个聚类中心中选取，而这些聚类中心的取值在Index训练后就固定了，于是，我们可以预先计算相同列的子向量之间的距离，保存在内存中。在检索时，将query向量做乘积量化后，直接查表就可以得到子向量之间的L2距离的平方，最后将子向量加在一起，就得到了query向量与物料向量的L2距离的平方

**非对称检索**:  
举个简单的例子:假定 M = 2，ksub = 4，即把向量分为两列，每列有四个聚类中心。压缩后的物料向量可能像这个样子[0,3]，其中0，1，2，3分别代表了对应列子向量聚类中心的向量。在得到query向量后，我们需要把query向量也切成两列，计算切分后的子向量与子向量聚类中心的距离，最后得到距离表dis_table如下
<div align=center><img src="https://github.com/AirGalaxy/faiss_note/blob/main/drawio/PQ4.drawio.png?raw=true"></div>
解释下这个表，第一行第一列代表query向量的第一列子向量与第一列子向量的第一个聚类中心的距离，其他以此类推。这样，query与物料向量[0，3]的距离应该为8+7=15。  

注意一点，每次一个新的query向量到来时，我们都要重新计算dis_tables，这是显然的

**对称检索**  
同样的，假定M = 2， ksub = 4，在query来之前，我们就可以预先计算好子向量聚类中心之间的距离:
<div align=center><img src="https://github.com/AirGalaxy/faiss_note/blob/main/drawio/PQ5.drawio.png?raw=true"></div>
解释下这个矩阵的含义，m=1代表这是切分后第一列子向量，表中第m行第n列，代表第一列子向量的第m个聚类中心与第n个聚类中心的距离的平方。于是可以看到这个矩阵是对称的(m与n的距离等于n与m的距离)，且对角线上的元素最大比其所在行和列的元素都要大(对于L2距离来说，自己与自己的距离是最大的)。   

举例来说，对于一个query向量，首先调用compute_code计算出query向量乘积量化的结果[1,2]，同样计算其与物料向量[0,3]的距离，在m=1的矩阵中找到(1,0)位置上的值为6，在m=2的矩阵上找到(2,3)位置上的值为10，因此query向量[1,2]与物料向量[0,3]的距离的平方为6+10=16

**对称与非对称检索的优劣**  
由于非对称检索只进行了一次量化，所以检索精度会更高。对称检索需要对query向量做量化，但是后续查表只需要查一张表，在多个query进行检索时，只需要存储一张dis_table，所占用的内存较小

## 3. ProductQuantizer类
接下来介绍PQ类的最后一类方法:检索方法 
### 非对称检索 
下面这个方法提供了非对称检索的实现
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
                //每个子向量占用的bit数
                nbits,
                //物料向量编码后的数组
                codes,
                //物料向量的个数
                ncodes,
                //query子向量到子向量聚类中心距离表
                dis_table,
                //搜索最近邻的k个
                k,
                //存放距离的堆数组
                heap_dis,
                //存放id的堆数组
                heap_ids);
        
        if (init_finalize_heap) {
            //原地做堆排序
            heap_reorder<C>(k, heap_dis, heap_ids);
        }

    }
 }
```
pq_knn_search_with_tables中按照query粒度去并行，处理单个query的靠pq_estimators_from_tables_generic，代码如下:
```c++
template <class C>
void pq_estimators_from_tables_generic(
        const ProductQuantizer& pq,
        size_t nbits,
        const uint8_t* codes,
        size_t ncodes,
        const float* dis_table,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids) {
    const size_t M = pq.M;
    const size_t ksub = pq.ksub;
    //按照物料向量来遍历
    for (size_t j = 0; j < ncodes; ++j) {
        //codes + j * pq.code_size: 指针移动到第j个物料向量起始编码位置
        PQDecoderGeneric decoder(codes + j * pq.code_size, nbits);
        float dis = 0;
        const float* __restrict dt = dis_table;
        for (size_t m = 0; m < M; m++) {
            //从物料向量编码中解出第m列子向量聚类中心的编码
            uint64_t c = decoder.decode();
            //dt[c]: 取到第m列的query子向量到当前物料向量在第m列上的子向量的距离平方
            //将子向量的距离平方加入到dis中
            dis += dt[c];
            //移动到同一个物料向量的下一列的子向量的距离表的起始位置
            dt += ksub;
        }
        //遍历完所有子向量后，得到当前物料向量与query向量的距离的平方dis
        if (C::cmp(heap_dis[0], dis)) {
            //如果dis比最小的元素大，把最小的元素用当前的dis替换掉
            heap_replace_top<C>(k, heap_dis, heap_ids, dis, j);
        }
    }
}
```
pq_estimators_from_tables_generic实现了上文所述的非对称检索，步骤和我们在预备知识中分析的完全相同

### 对称检索
ProductQuantizer类提供了对称检索的接口search_sdc。不过我们首先看下计算对称距离表的compute_sdc_table方法:
```c++
void ProductQuantizer::compute_sdc_table() {
    sdc_table.resize(M * ksub * ksub);
    ...
    #pragma omp parallel for
        for (int m = 0; m < M; m++) {
            //取第m列的聚类中心的起始值
            const float* cents = centroids.data() + m * ksub * dsub;
            //取sdc_table的第m列的聚类中心的存放结果的起始地址
            float* dis_tab = sdc_table.data() + m * ksub * ksub;
            pairwise_L2sqr(
                    //子聚类中心的维度
                    dsub, 
                    //我们要计算多少个子聚类中心到相同列子聚类中心的距离？
                    ksub,
                    //第m列子聚类中心的起始地址
                    cents,
                    //第m列子聚类中心的个数
                    ksub, 
                    //所有子聚类中心的起始地址
                    cents, 
                    //存放结果的数组
                    dis_tab, 
                    dsub, dsub, ksub);
        }
}
```
可以看到，计算对称距离表和在预备知识中分析的完全一致，按照子向量的维度进行并行的处理。在相同列中的所有子聚类中心计算其两两之间的距离并保存到dis_tab中，上文对pairwise_L2sqr已经有过分析，这里不再赘述

现在我们看下search_sdc的实现:
```c++
void ProductQuantizer::search_sdc(
        const uint8_t* qcodes,
        size_t nq,
        const uint8_t* bcodes,
        const size_t nb,
        float_maxheap_array_t* res,
        bool init_finalize_heap) const {
    //要搜索最近的k个向量
    size_t k = res->k;

#pragma omp parallel for
    for (int64_t i = 0; i < nq; i++) {
        //移动堆数组到第i个向量检索结果的位置
        idx_t* heap_ids = res->ids + i * k;
        float* heap_dis = res->val + i * k;
        //移动query向量编码到第i个向量的位置
        const uint8_t* qcode = qcodes + i * code_size;
        //初始化堆
        if (init_finalize_heap)
            maxheap_heapify(k, heap_dis, heap_ids);

        const uint8_t* bcode = bcodes;
        //遍历物料向量
        for (size_t j = 0; j < nb; j++) {
            float dis = 0;
            const float* tab = sdc_table.data();
            //遍历子向量
            for (int m = 0; m < M; m++) {
                //从距离表中查出第m列子向量上query向量与第j个物料向量的距离
                dis += tab[bcode[m] + qcode[m] * ksub];
                tab += ksub * ksub;
            }
            //插入堆中
            if (dis < heap_dis[0]) {
                maxheap_replace_top(k, heap_dis, heap_ids, dis, j);
            }
            //移动到下一个物料向量
            bcode += code_size;
        }
        //堆排序
        if (init_finalize_heap)
            maxheap_reorder(k, heap_dis, heap_ids);
    }
}
```
很朴素的实现，openMP按照query粒度并行，查表计算距离+堆排序，和预备知识中的实现相同
