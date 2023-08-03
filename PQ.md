# Fassi源码阅读
本节为乘积量化方法的预备知识
## 1. 什么是乘积量化方法
还记得前面的IVF方法吗，里面提到量化的一个想法就是用把物料向量量化到聚类中心的向量上，但是这样做的误差太大了。可能我们有100万个向量，但是我们的聚类中心只有65536个，用65535个向量来表示100万个向量显然是不合适的。  
有什么能扩展量化结果的可能性的方法呢？我们想到我们可以先对原始向量进行切分，然后对切分后的子向量做聚类，用子向量的聚类中心组合来做为量化结果。对于一个1024维的向量，我们把1-512维切分为子向量1，513-1024维切分为子向量2，每个子向量(65536/2)个聚类中心，现在我们可以用这些聚类中心表示32768*32768=1073741824个向量。  
可以看到使用了相同数量的聚类中心，能表示的向量却多了很多。

假设我们的物料向量有$M$个，每个物料向量有$d$维。切分的子向量数量为N。PQ方法的步骤如下:
1. 切分子向量  
如下图所示，我们将原始的向量切分为了N个子向量，每个子向量$d/N$维 
<div align=center><img src="https://github.com/AirGalaxy/fassi_note/blob/main/drawio/PQ1.drawio.png?raw=true" width="75%"></div>


2. 子向量聚类  
我们对子向量聚类，得到每一列子向量的聚类中心，假设一列子向量有C个聚类中心，我们对C个聚类中心进行编号1-C，记录编号到实际聚类中心的映射，称为codebook。如下图所示:

<div align=center><img src="https://github.com/AirGalaxy/fassi_note/blob/main/drawio/pq2.drawio.png?raw=true" width="75%"></div>

3. 向量量化  
我们对所有的子向量做聚类操作，得到所有子向量的codebook，如下图所示:
<div align=center><img src="https://github.com/AirGalaxy/fassi_note/blob/main/drawio/pq3.jpg?raw=true" width="75%"></div>

如上图所示，我们得到了N个codebook。将每个子向量用其所属聚类中心的编号代替，这就是量化的过程。如第一个向量可以被量化为(2,43,...,87)

## 2. Quantizer类
量化器接口类  
*~~另外吐槽下，Level1Quantizer名字带Quantizer，做的也是量化器该做的事，但却不是Quantizer的子类~~*
```c++
struct Quantizer {
    // 每个向量的维度
    size_t d;
    // 每个被索引的向量占多数个字节        
    size_t code_size;

    explicit Quantizer(size_t d = 0, size_t code_size = 0)
            : d(d), code_size(code_size) {}

    /** Train the quantizer
     *
     * @param x       training vectors, size n * d
     */
    virtual void train(size_t n, const float* x) = 0;

    /** Quantize a set of vectors
     *
     * @param x        input vectors, size n * d
     * @param codes    output codes, size n * code_size
     */
    virtual void compute_codes(const float* x, uint8_t* codes, size_t n)
            const = 0;

    /** Decode a set of vectors
     *
     * @param codes    input codes, size n * code_size
     * @param x        output vectors, size n * d
     */
    virtual void decode(const uint8_t* code, float* x, size_t n) const = 0;

    virtual ~Quantizer() {}
};
```
源码写的很清楚，首先这个Quantizer就是为了乘积量化。如果你看过Level1Quantizer的情况下，这些字段的含义很明确。
* trian() 训练量化器，找到聚类中心
* compute_codes() 把对输入向量集合x进行量化，量化后的结果放到codes中
* decode() 把量化后的编码codes解码为原始向量，结果放入到x中

## 3. ProductQuantizer
仅为L2距离实现的乘积量化器
## 3.1 成员变量
```c++
struct ProductQuantizer : Quantizer {
    //子量化器的数量(子量化器可以理解为Level1Quantizer)
    size_t M;     
    //每个量化后的索引占多少bit
    size_t nbits; 

    //子向量的维数
    size_t dsub; 
    //子向量的聚类时的聚类中心的个数
    size_t ksub; 

    enum train_type_t {
        //默认训练策略
        Train_default,
        //训练时的聚类中心已经初始化
        Train_hot_start,     
        //不同的PQ分段共享词典
        Train_shared,        
        //使用超立方体采样初始化聚类中心
        Train_hypercube,
        //使用超立方体采样+pca初始化聚类中心    
        Train_hypercube_pca
    };
    train_type_t train_type;
    // 聚类策略
    ClusteringParameters cp;
    // 使用这个Index来寻找子聚类中心
    Index* assign_index;
    // 聚类中心表 布局是M * ksub *dsub
    std::vector<float> centroids;
    // 转置后的聚类中心表，布局是dsub * M * ksub
    std::vector<float> transposed_centroids;
    // 存储了子向量聚类中心的向量的L2范数的平方
    std::vector<float> centroids_sq_lengths;
}
```
注意下centroids，要从centroids中取到指定的位置上的聚类中心向量。第m个codebook的第k个聚类中心的向量地址应该为[centroids + dsub * ksub * m + k * dsub, centroids + dsub * ksub * m + (k + 1) * dsub]，可以参考下面的读取聚类中心的方法:
```c++
    // 读取编号为m的codebook的第i个聚类中心向量
    float* get_centroids(size_t m, size_t i) {
        return &centroids[(m * ksub + i) * dsub];
    }
    const float* get_centroids(size_t m, size_t i) const {
        return &centroids[(m * ksub + i) * dsub];
    }
```
### 3.2 成员函数
先看下构造函数:
```c++
ProductQuantizer::ProductQuantizer(
    //原始输入向量的维数
    size_t d, 
    //要分割成多少个子向量
    size_t M, 
    //每个子向量的索引占多少个bit
    //默认子向量的索引占的bit数所能代表的最大数量为子向量聚类中心的个数
    size_t nbits)
        //Quantizer中的code_size在set_derived_values中计算
        : Quantizer(d, 0), M(M), nbits(nbits), assign_index(nullptr) {
    set_derived_values();
}
void ProductQuantizer::set_derived_values() {
    //每个子向量多少维
    dsub = d / M;
    //量化后的物料向量占多少字节
    code_size = (nbits * M + 7) / 8;
    //每个子量化器有多少个聚类中心
    ksub = 1 << nbits;
    //预分配空间，一共dsub * ksub * M = d * ksub个float
    centroids.resize(d * ksub);
    //默认训练方式
    train_type = Train_default;
}
```
构造函数中就是通过元素向量的维数、子向量聚类中心的个数，算出每个子向量多少维，量化编码后的物料向量占多少个字节，然后为保存子聚类中心的容器分配空间

设置第m个codebook的向量
```c++
void ProductQuantizer::set_params(const float* centroids_, int m) {
    memcpy(get_centroids(m, 0),
           centroids_,
           ksub * dsub * sizeof(centroids_[0]));
}
```
#### 3.2.1 训练函数
和Level1Quantizer类似，ProductQuantizer也是要先训练，才能做量化的，我们按这个顺序，先看下训练的过程。
* 所有的分段都共享相同的聚类中心，即train_type == Train_shared的情况
```c++
void ProductQuantizer::train(size_t n, const float* x) {
    if (train_type == Train_shared) {
        Clustering clus(dsub, ksub, cp);
        IndexFlatL2 index(dsub);
        // 把所有的子向量全部送去进行kmeans聚类
        clus.train(n * M, x, assign_index ? *assign_index : index);
        for (int m = 0; m < M; m++) {
            //M段子向量的聚类中心(codebook)都是一样的
            set_params(clus.centroids.data(), m);
        }
    }
}
```
这种情况比较简单，具体的情况已经在Level1Quantizer中分析过

* train_type为其他值的情况
```c++
    final_train_type = train_type;
    if (train_type == Train_hypercube ||
        train_type == Train_hypercube_pca) {
        if (dsub < nbits) {
            final_train_type = Train_default;
        }
    }
```
如果每个子向量的维度比量化后的向量的所占的bit数还少，就不进行hypercube初始化。

```c++
    //放一列子向量的缓存空间
    float* xslice = new float[n * dsub];
    ScopeDeleter<float> del(xslice);

    for (int m = 0; m < M; m++) {
        //重整输入向量形状，取第m列所有子向量放到xslice中
        for (int j = 0; j < n; j++)
            memcpy(xslice + j * dsub,
                   x + j * d + m * dsub,
                   dsub * sizeof(float));
        Clustering clus(dsub, ksub, cp);

        // we have some initialization for the centroids
        if (final_train_type != Train_default) {
            //预分配空间，用来放入初始化的聚类中心
            clus.centroids.resize(dsub * ksub);
        }

        switch (final_train_type) {
            case Train_hypercube:
                // 超立方体初始化聚类中心
                init_hypercube(
                        dsub, nbits, n, xslice, clus.centroids.data());
                break;
                // 超立方体+pca初始化聚类中心
            case Train_hypercube_pca:
                init_hypercube_pca(
                        dsub, nbits, n, xslice, clus.centroids.data());
                break;
                // 热启动方式，初始聚类中心由外部传入
            case Train_hot_start:
                memcpy(clus.centroids.data(),
                       get_centroids(m, 0),
                       dsub * ksub * sizeof(float));
                break;
            default:;
        }
        //聚类，把聚类中心放到成员变量centroids中
        IndexFlatL2 index(dsub);
        clus.train(n, xslice, assign_index ? *assign_index : index);
        set_params(clus.centroids.data(), m);
    }
```
可以看到除了Train_shared的情况，其他的都基本一致，只是在初始化聚类中心的策略不同。

#### 3.2.2 编码(量化)函数
先看下对单个向量进行编码的函数:
```c++
void ProductQuantizer::compute_code(const float* x, uint8_t* code) const {
    switch (nbits) {
        case 8:
            faiss::compute_code<PQEncoder8>(*this, x, code);
            break;

        case 16:
            faiss::compute_code<PQEncoder16>(*this, x, code);
            break;

        default:
            faiss::compute_code<PQEncoderGeneric>(*this, x, code);
            break;
    }
}
```
这里实现都是由模板函数compute_code实现的，同时对子向量量化后所占位数nbits = 8，nbits = 16的情况做了优化。
简单看下compute_code的方法:
```c++
template <class PQEncoder>
void compute_code(const ProductQuantizer& pq, const float* x, uint8_t* code) {
    // 预先分配了一个数组来记录所有子向量聚类中心到要量化的向量的距离
    // 下面会解释为什么这么做
    std::vector<float> distances(pq.ksub);
    // 定义encoder
    PQEncoder encoder(code, pq.nbits);
    // 对于第m列子向量
    for (size_t m = 0; m < pq.M; m++) {
        //要编码向量的第m列子向量
        const float* xsub = x + m * pq.dsub;

        uint64_t idxm = 0;
        if (pq.transposed_centroids.empty()) {
            // 寻找最近的子向量聚类中心
            idxm = fvec_L2sqr_ny_nearest(
                    //存放距离的数组
                    distances.data(),
                    //要编码的子向量起始地址
                    xsub,
                    // codebook
                    pq.get_centroids(m, 0),
                    // 子向量维度
                    pq.dsub,
                    // 子向量个数
                    pq.ksub);
        } else {
            // transposed centroids are available, use'em
            idxm = fvec_L2sqr_ny_nearest_y_transposed(
                    distances.data(),
                    xsub,
                    pq.transposed_centroids.data() + m * pq.ksub,
                    pq.centroids_sq_lengths.data() + m * pq.ksub,
                    pq.dsub,
                    pq.M * pq.ksub,
                    pq.ksub);
        }

        encoder.encode(idxm);
    }
}
```
代码的逻辑和我们上面说的PQ方法完全一致，找到最近的子向量聚类中心对应的id，然后对id进行编码。注意这里找最近的子向量并没有像Level1Quantizer一样，而是使用fvec_L2sqr_ny_nearest()去寻找。接下来我们就看下fvec_L2sqr_ny_nearest是怎么工作的。
```c++
void fvec_L2sqr_ny_ref(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    //直接循环计算向量之间的距离
    for (size_t i = 0; i < ny; i++) {
        //fvec_L2sqr:计算两个维度为d的向量之间的距离
        //我们在Level1Quantizer中分析过
        dis[i] = fvec_L2sqr(x, y, d);
        y += d;
    }
}

size_t fvec_L2sqr_ny_nearest_ref(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    //先计算子聚类中心到要编码的子向量的距离，放入distances_tmp_buffer数组中
    fvec_L2sqr_ny(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;
    //寻找最小距离的id
    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }
    return nearest_idx;
}
```
代码很简单，不用我解释大家也懂。  
不过这里有个问题，为什么我们要用distances_tmp_buffer这个数组去存储所有的聚类中心到子向量的距离呢？我们只需要记录最近的距离就行了。比如可以写出如下简化的代码:
```c++
    size_t nearest_idx = 0;
    float min_dis = HUGE_VALF;
    //寻找最小距离的id
    for (size_t i = 0; i < ny; i++) {
        float current_dis = fvec_L2sqr(x, y, d);
        if （current_dis < min_dis） {
            min_dis =current_dis;
            nearest_idx = i;
        }
        y += d;
    }
    return nearest_idx;
```
是的，这是完全正确的，但是fassi中的注释提到，设置distances_tmp_buffer是为了编译器可以进行向量化的优化，我们"简化"后的代码生成的CPU指令远远多于( significantly more than)使用distances_tmp_buffer数组，实际上更慢了。
