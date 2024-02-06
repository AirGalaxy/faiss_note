# Faiss源码阅读
## PQ检索的实现——IndexPQ类
IndexPQ类是对PQ方法的一个实现，在介绍了ProductQuantizer类之后，IndexPQ类就很容易理解了
简单看下IndexPQ的成员变量
```c++
struct IndexPQ : IndexFlatCodes {
    //上几节提到的乘积量化器
    ProductQuantizer pq;
    // 是否要做多义性编码训练
    // 这里我也没明白原理是什么，不过我会介绍这种情况下训练和搜索的过程
    bool do_polysemous_training; ///< false = standard PQ

    //上文说的多义性搜索的汉明门限
    int polysemous_ht;

    //检索的类型
    Search_type_t search_type;

}
```
这里提一下Search_type_t，这是一个枚举，包含以下枚举项:
```c++
    /// how to perform the search in search_core
    enum Search_type_t {
        //非对称的PQ检索，这也是默认的检索方式
        ST_PQ,                    
        //汉明距离作为距离的检索方式
        ST_HE,                    ///< Hamming distance on codes
        ST_generalized_HE,        ///< nb of same codes
        //PQ对称检索，精度比ST_PQ低，但是速度更快
        ST_SDC,                   ///< symmetric product quantizer (SDC)
        //使用汉明距离过滤，结合PQ方法的检索
        ST_polysemous,            ///< HE filter (using ht) + PQ combination
        ST_polysemous_generalize, ///< Filter on generalized Hamming
    };
```
为什么可以使用汉明距离作为衡量两个向量的距离？在接下来的训练过程中我会解释

## IndexPQ的训练过程
训练过程的入口在train()中:
```c++
void IndexPQ::train(idx_t n, const float* x) {
    if (!do_polysemous_training) { 
        // 不做多义性训练，那么直接调用ProductQuantizer进行标准的训练
        pq.train(n, x);
    } else {
        // 我们要选取多少个点来进行训练？
        idx_t ntrain_perm = polysemous_training.ntrain_permutation;

        if (ntrain_perm > n / 4)
            ntrain_perm = n / 4;
        //从训练集中抽取前(n - ntrain_perm)个元素来训练乘积量化器
        pq.train(n - ntrain_perm, x);
        //使用模拟退火算法，调整pq中聚类中心的位置
        polysemous_training.optimize_pq_for_hamming(
                pq, ntrain_perm, x + (n - ntrain_perm) * d);
    }
    is_trained = true;
}
```
这段代码中，pq.train(n,x)我们在上面的章节中已经分析过。那``polysemous_training.optimize_pq_for_hamming``是在干什么呢？参考下面的代码:
```c++
void PolysemousTraining::optimize_reproduce_distances(
        ProductQuantizer& pq) const {
#pragma omp parallel for num_threads(nt)
    // 对于PQ量化后的每一维子向量数据
    for (int m = 0; m < pq.M; m++) {
        std::vector<double> dis_table;
        //取该维数据的聚类中心
        float* centroids = pq.get_centroids(m, 0);
        //计算聚类中心之间的距离，得到距离表
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dis_table.push_back(fvec_L2sqr(
                        centroids + i * dsub, centroids + j * dsub, dsub));
            }
        }

        std::vector<int> perm(n);
        //定义了优化过程中损失函数的计算和更新过程
        ReproduceWithHammingObjective obj(nbits, dis_table, dis_weight_factor);
        //使用模拟退火优化，计算聚类中心的排列，结果放在perm中
        SimulatedAnnealingOptimizer optim(&obj, *this);
                std::vector<float> centroids_copy;
        
        for (int i = 0; i < dsub * n; i++)
            centroids_copy.push_back(centroids[i]);
        //根据优化后的结果perm，调整聚类中心的顺序
        for (int i = 0; i < n; i++)
            memcpy(centroids + perm[i] * dsub,
                   centroids_copy.data() + i * dsub,
                   dsub * sizeof(centroids[0]));
    }
}
```

我们只需要分析损失函数，这个函数在ReproduceWithHammingObjective类中，参数perm为对训练样本(这里就是PQ后的聚类中心)重新排列后的index列表，通过这个函数就可以知道优化的目的:
```c++
    // cost = quadratic difference between actual distance and Hamming distance
    double compute_cost(const int* perm) const override {
        //初始化cost为0
        double cost = 0;
        //n = 1 << nbits, nbits为pq中，物料向量的子向量量化后所占的bit数
        //遍历PQ一个子向量的所有聚类中心之间的距离
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                //距离表上第(i, j)位置上标准化到汉明距离上的向量距离
                double wanted = target_dis[i * n + j];
                //第(i, j)位置上距离的权重
                double w = weights[i * n + j];
                //在排列perm的情况下，第i个向量与第j个向量的汉明距离
                double actual = hamming_dis(perm[i], perm[j]);
                //标准化后的向量距离与汉明距离的差
                cost += w * sqr(wanted - actual);
            }
        }
        return cost;
    }
```
我们希望找到一个排列perm，使得cost最少，在这种情况下，perm对应的汉明距离与向量的距离最少    
eg. 开始时样本的index为0,1,2，对应的向量为v1,v2,v3，找到向量的index的排列，如(2,0,1),通过计算上面的cost，找到排列最小的cost

这里的target_dis和weights是怎么来的呢？在构造函数中通过pq的聚类中心的距离表dis_table进行映射计算得到
```c++
    void set_affine_target_dis(const std::vector<double>& dis_table) {
        double sum = 0, sum2 = 0;
        int n2 = n * n;
        for (int i = 0; i < n2; i++) {
            //距离表中所有元素之和
            sum += dis_table[i];
            //计算距离表中所有元素平方之和
            sum2 += dis_table[i] * dis_table[i];
        }
        //计算所有距离的平均值
        double mean = sum / n2;
        //计算标准差
        double stddev = sqrt(sum2 / n2 - (sum / n2) * (sum / n2));
        
        target_dis.resize(n2);
        //计算target_dis
        for (int i = 0; i < n2; i++) {
            // 映射函数，其实就是z-score标准化，将距离映射到（0 - nbits）上
            double td = (dis_table[i] - mean) / stddev * sqrt(nbits / 4) +
                    nbits / 2;
            target_dis[i] = td;
            // 计算距离对应的权重
            weights.push_back(dis_weight(td));
        }
    }
```

计算距离的权重:
```c++
    // 对于较大的距离x，权重较小，对于较小的权重x，权重较大
    // 默认dis_weight_factor = log(2)
    double dis_weight(double x) const {
        return exp(-dis_weight_factor * x);
    }
```
总的来说，target_dis与weights就是对原始的距离表做了一次标准化，将距离映射到(0 - nbits)上。  
那么为什么要映射到0-nbits上？可以想到，距离之间的汉明距离最少是0，最长是nbits，如果我们要通过汉明距离来衡量向量的距离，那么在优化时，必须要把向量的距离标准化到汉明距离的范围上。  


现在我们对这里的优化有了一个认识:优化后,在PQ后的一维子向量的所有聚类中心中,他们在pq.centroids的保存顺序的编号(0,1,2,...,ksub)的汉明距离，与其编号对应的向量距离(如L2距离)，其差值之和是最小的，也就是说，**<span style="color:green;">我们可以用聚类中心编号的汉明距离来衡量聚类中心向量之间的向量距离</span>**。

## IndexPQ的搜索过程
```c++
void IndexPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* iparams) const {
    search_type = params->search_type;        
    //search_type为PQ检索
    if (search_type == ST_PQ) { // Simple PQ search
        if (metric_type == METRIC_L2) {
            //初始化结果堆
            float_maxheap_array_t res = {
                    size_t(n), size_t(k), labels, distances};
            //委托给PQ去检索
            pq.search(x, n, codes.data(), ntotal, &res, true);
        } else {
            float_minheap_array_t res = {
                    size_t(n), size_t(k), labels, distances};
            //PQ检索，使用内积计算距离
            pq.search_ip(x, n, codes.data(), ntotal, &res, true);
        }
        indexPQ_stats.nq += n;
        indexPQ_stats.ncode += n * ntotal;

    } else if (
            //search_type为多义性检索
            search_type == ST_polysemous ||
            search_type == ST_polysemous_generalize) {
       // 汉明距离的门限，汉明距离小于polysemous_ht认为相似
        int polysemous_ht =
                params ? params->polysemous_ht : this->polysemous_ht;
        //多义性检索，基于汉明距离
        search_core_polysemous(
                n,
                x,
                k,
                distances,
                labels,
                polysemous_ht,
                search_type == ST_polysemous_generalize);

    } else {
        ...
    }
}
```
接下来我们要分析一下search_core_polysemous:
```c++
void IndexPQ::search_core_polysemous(
    //query向量的个数
    idx_t n,
    //query向量的首地址
    const float* x,
    //每个query搜索结果取topk个
    idx_t k,
    //存放query向量到
    float* distances,
    //存放搜索结果的id
    idx_t* labels,
    //汉明门限
    int polysemous_ht,
    bool generalized_hamming) const {

    //如果门限为0，则门限开放到最大
    if (polysemous_ht == 0) {
        //pq.nbits * pq.M为一个完整的物料向量量化后所占的bit数
        //任何query向量到物料向量的汉明距离都不会超过pq.nbits * pq.M + 1
        polysemous_ht = pq.nbits * pq.M + 1;
    }

    //计算所有query向量到物料向量的距离
    //注意这里并没有对query向量先做PQ再计算距离
    float* dis_tables = new float[n * pq.ksub * pq.M];
    ScopeDeleter<float> del(dis_tables);
    pq.compute_distance_tables(n, x, dis_tables);

    //将query做PQ量化，得到的PQ编码q_codes就是计算汉明距离的编码
    uint8_t* q_codes = new uint8_t[n * pq.code_size];
    ScopeDeleter<uint8_t> del2(q_codes);
#pragma omp parallel for
    for (idx_t qi = 0; qi < n; qi++) {
        pq.compute_code_from_distance_table(
                dis_tables + qi * pq.M * pq.ksub,
                q_codes + qi * pq.code_size);
    }
    size_t n_pass = 0;

    int bad_code_size = 0;

#pragma omp parallel for reduction(+ : n_pass, bad_code_size)
    for (idx_t qi = 0; qi < n; qi++) {
        //取第qi个query对应的汉明编码
        const uint8_t* q_code = q_codes + qi * pq.code_size;
        //qi个query到聚类中心的距离
        const float* dis_table_qi = dis_tables + qi * pq.M * pq.ksub;
        //初始化存放结果的堆
        int64_t* heap_ids = labels + qi * k;
        float* heap_dis = distances + qi * k;
        maxheap_heapify(k, heap_dis, heap_ids);

        if (!generalized_hamming) {
            switch (pq.code_size) {
                case 4:
                    // 搜索一个query向量
                    n_pass += polysemous_inner_loop<HammingComputer4>(
                            *this,
                            dis_table_qi,
                            q_code,
                            k,
                            heap_dis,
                            heap_ids,
                            polysemous_ht);
                    break;
                case 8:
                    ...
                case 16:
                    ...
                case 20:
                    ...
                default:
                    if (pq.code_size % 4 == 0) {
                        n_pass += polysemous_inner_loop<HammingComputerDefault>(
                                *this,
                                dis_table_qi,
                                q_code,
                                k,
                                heap_dis,
                                heap_ids,
                                polysemous_ht);
                    } else {
                        bad_code_size++;
                    }
                    break;
            }
        } else {
            //逻辑与上面类似，不赘述
        }
        //堆排序
        maxheap_reorder(k, heap_dis, heap_ids);
    }
}
```
主要检索由polysemous_inner_loop完成，polysemous_inner_loop实现了对一个query向量进行检索的过程:
```c++
template <class HammingComputer>
static size_t polysemous_inner_loop(
        const IndexPQ& index,
        const float* dis_table_qi,
        const uint8_t* q_code,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        int ht) {
    //划分为了M个子向量
    int M = index.pq.M;
    //PQ量化后一个向量占用的bit数
    int code_size = index.pq.code_size;
    //子向量维度有ksub个子量化器
    int ksub = index.pq.ksub;
    //ntotal个向量
    size_t ntotal = index.ntotal;
    //物料向量首地址
    const uint8_t* b_code = index.codes.data();

    size_t n_pass_i = 0;
    //汉明距离计算器
    HammingComputer hc(q_code, code_size);

    for (int64_t bi = 0; bi < ntotal; bi++) {
        //计算query向量对物料向量的汉明距离
        int hd = hc.hamming(b_code);
        //根据汉明门限过滤
        if (hd < ht) {
            //记录超过汉明门限的物料向量的个数
            n_pass_i++;

            float dis = 0;
            const float* dis_table = dis_table_qi;
            //计算query向量到物料向量的实际距离(L2距离？)
            for (int m = 0; m < M; m++) {
                dis += dis_table[b_code[m]];
                dis_table += ksub;
            }
            //距离更近，把堆中最差的结果淘汰出去
            if (dis < heap_dis[0]) {
                maxheap_replace_top(k, heap_dis, heap_ids, dis, bi);
            }
        }
        //指针指到下一个物料向量
        b_code += code_size;
    }
    //返回没有被汉明距离过滤的物料向量的个数
    return n_pass_i;
}
```
## 总结
IndexPQ支持两种检索方式
1. 朴素的PQ检索功能，这个功能由ProductQuantizer类实现
2. search_core_polysemous检索方式中，使用汉明距离过滤掉一部分物料向量，再进行PQ的非对称检索。