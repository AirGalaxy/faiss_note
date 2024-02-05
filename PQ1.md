# Faiss源码阅读
本节为乘积量化方法的预备知识
## 1. 什么是乘积量化方法
还记得前面的IVF方法吗，里面提到量化的一个想法就是用把物料向量量化到聚类中心的向量上，但是这样做的误差太大了。可能我们有100万个向量，但是我们的聚类中心只有65536个，用65535个向量来表示100万个向量显然是不合适的。  
有什么能扩展量化结果的可能性的方法呢？我们想到我们可以先对原始向量进行切分，然后对切分后的子向量做聚类，用子向量的聚类中心组合来做为量化结果。对于一个1024维的向量，我们把1-512维切分为子向量1，513-1024维切分为子向量2，每个子向量(65536/2)个聚类中心，现在我们可以用这些聚类中心表示32768*32768=1073741824个向量。  
可以看到使用了相同数量的聚类中心，能表示的向量却多了很多。

假设我们的物料向量有$M$个，每个物料向量有$d$维。切分的子向量数量为N。PQ方法的步骤如下:
1. 切分子向量  
如下图所示，我们将原始的向量切分为了N个子向量，每个子向量$d/N$维 
<div align=center><img src="https://github.com/AirGalaxy/faiss_note/blob/main/drawio/PQ1.drawio.png?raw=true" width="75%"></div>


2. 子向量聚类  
我们对子向量聚类，得到每一列子向量的聚类中心，假设一列子向量有C个聚类中心，我们对C个聚类中心进行编号1-C，记录编号到实际聚类中心的映射，称为codebook。如下图所示:

<div align=center><img src="https://github.com/AirGalaxy/faiss_note/blob/main/drawio/pq2.drawio.png?raw=true" width="75%"></div>

3. 向量量化  
我们对所有的子向量做聚类操作，得到所有子向量的codebook，如下图所示:
<div align=center><img src="https://github.com/AirGalaxy/faiss_note/blob/main/drawio/pq3.jpg?raw=true" width="75%"></div>

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

## 3. ProductQuantizer类
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
代码的逻辑和我们上面说的PQ方法完全一致，找到最近的子向量聚类中心对应的id，然后对id进行编码。注意这里找最近的子向量并没有像Level1Quantizer一样，而是使用fvec_L2sqr_ny_nearest()去寻找。通过名字就知道这是用L2距离的平方来当作距离远近。  
接下来我们就看下fvec_L2sqr_ny_nearest是怎么工作的。
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
是的，这是完全正确的，但是faiss中的注释提到，设置distances_tmp_buffer是为了编译器可以进行向量化的优化，我们"简化"后的代码生成的CPU指令远远多于( significantly more than)使用distances_tmp_buffer数组，实际上更慢了。

当聚类中心的矩阵有转置形式时，计算矩阵的过程又略有不同
```c++
    idxm = fvec_L2sqr_ny_nearest_y_transposed(
            //存放距离的数组，作用如上
            distances.data(),
            //要计算到聚类中心距离的子向量
            xsub,
            //转置后聚类中心矩阵的第m列子向量的起始地址
            pq.transposed_centroids.data() + m * pq.ksub,
            //第m列子向量聚类中心的L2范数的平方起始位置
            pq.centroids_sq_lengths.data() + m * pq.ksub,
            //子向量维数
            pq.dsub,
            //对于转置后聚类中心矩阵，对于聚类中心向量的第i维到第i+1维，要前进多少个float的距离 
            pq.M * pq.ksub,
            //聚类中心的个数
            pq.ksub);
```
参数被透传到如下函数:
```c++
void fvec_L2sqr_ny_y_transposed_ref(
        float* dis,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        //先计算要编码的子向量的L2范数的平方
        x_sqlen += x[j] * x[j];
    }
    //依次遍历聚类中心
    for (size_t i = 0; i < ny; i++) {
        float dp = 0;
        for (size_t j = 0; j < d; j++) {
            //计算完全平方和中的交叉项
            dp += x[j] * y[i + j * d_offset];
        }
        //减去交叉项，就得到L2距离
        dis[i] = x_sqlen + y_sqlen[i] - 2 * dp;
    }
}
```
$$||a-b||^2 = ||a||^2 + ||b||^2 - \sum_{i=1}^d{a_i*b_i} $$
其中$||a||$代表向量a的L2距离。很简单的推导，拆开用完全平方公式加一下就知道了。
~~但为什么要这样做呢，求大佬解惑~~  

我们回到compute_code的代码，现在的代码类似这样:
```c++
    PQEncoder encoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++) {
     ...
     //已经得到最近的子向量聚类中心编号idmx
        encoder.encode(idxm);
    }
```

我们只需要看下PQEncoder的实现就行了
```c++
struct PQEncoder8 {
    uint8_t* code;
    PQEncoder8(uint8_t* code, int nbits);
    void encode(uint64_t x);
};
inline void PQEncoder8::encode(uint64_t x) {
    *code++ = (uint8_t)x;
}
```
直接把x的低8位放到code中，code在构造时由外部传入，作为输出参数。PQEncoder16实现完全一致。
通用的PQ编码器
```c++
inline void PQEncoderGeneric::encode(uint64_t x) {
    //offset为初始时在reg中的偏移，一般为0,reg为uint8_t
    //提取x中最后8位
    reg |= (uint8_t)(x << offset);
    //把放入reg中的部分移出x
    x >>= (8 - offset);
    //如果还需要编码的位数大于8，说明上面已经把reg填满了
    if (offset + nbits >= 8) {
        //把reg赋值给code，code指针向前移动
        *code++ = reg;
        //依次把x的中的数据一次8bit放入code中
        for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) {
            *code++ = (uint8_t)x;
            x >>= 8;
        }

        offset += nbits;
        //本次编码x还剩(offset&=7)个bits没有放入code中
        offset &= 7;
        //把x中最后不满8bit的数据放入reg中
        //等待下次调用encode或PQEncoderGeneric析构时把reg中数据放入codes
        reg = (uint8_t)x;
    } else {
        //子向量编码后不足8bit，直接更新offset就行了
        offset += nbits;
    }
}

inline PQEncoderGeneric::~PQEncoderGeneric() {
    //在encode中，如果offset！=0，就代表reg中还有数据没放入，在析构时放入code
    if (offset > 0) {
        *code = reg;
    }
}

```
可以看到上面编码的过程符合我们在第一小节中说的步骤:
1. 先找最近子向量聚类中心
2. 挨个将1-M个子聚类中心编号放入code中

上面compute_code函数的实现并没有使用assign_index去寻找最近的聚类中心，ProductQuantizer也提供了使用assign_index的版本:
```c++
void ProductQuantizer::compute_codes_with_assign_index(
        //要编码的向量集合
        const float* x,
        //存放编码结果的数组
        uint8_t* codes,
        //要编码的向量个数
        size_t n) {
    //按照每列子向量的顺序，每次处理所有要编码向量在第m列的子向量
    for (size_t m = 0; m < M; m++) {
        //先把assign_index清空
        assign_index->reset();
        //把当前列的子向量聚类中心加入index中
        assign_index->add(ksub, get_centroids(m, 0));

        //每次处理的分片大小
        size_t bs = 65536;
        float* xslice = new float[bs * dsub];
        ScopeDeleter<float> del(xslice);
        idx_t* assign = new idx_t[bs];
        ScopeDeleter<idx_t> del2(assign);

        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);
            //把分片内的要编码向量的m列子向量放入xslice中
            for (size_t i = i0; i < i1; i++) {
                //dest每次前进dsub，x每次前进d
                memcpy(xslice + (i - i0) * dsub,
                       x + i * d + m * dsub,
                       dsub * sizeof(float));
            }
            //检索，把最近的聚类中心的编号放入assign
            assign_index->assign(i1 - i0, xslice, assign);

            if (nbits == 8) {
                //第i0的第m列子向量的编码起始位置
                uint8_t* c = codes + code_size * i0 + m;
                for (size_t i = i0; i < i1; i++) {
                    //把聚类中心编号放入codes中
                    *c = assign[i - i0];
                    //走到下一个向量的同一列子向量的位置上
                    c += M;
                }
            } else if (nbits == 16) {
                //与nbits = 8 类似
                ...
            } else {
              //使用PQEncoderGenenric把assign编码到codes中
              ...
            }
        }
    }
}
```
compute_codes_with_assign_index提供了对n个向量进行编码的功能，在寻找最近的子向量聚类中心时，采用了assign_index来实现。其余编码过程与compute_code类似，对nbits=8、16做了优化，其余情况依然通过PQEncoderGenenric实现


此外，编码函数还有一个根据距离表进行量化的版本:
```c++
void ProductQuantizer::compute_code_from_distance_table(
        const float* tab,
        uint8_t* code) const {
    PQEncoderGeneric encoder(code, nbits);
    for (size_t m = 0; m < M; m++) {
        float mindis = 1e20;
        uint64_t idxm = 0;

        /* Find best centroid */
        for (size_t j = 0; j < ksub; j++) {
            float dis = *tab++;
            if (dis < mindis) {
                mindis = dis;
                idxm = j;
            }
        }

        encoder.encode(idxm);
    }
}
```
tab为要编码的向量的子向量到各个聚类中心的距离，tab的维度为 M * ksub。  
*(tab + m * ksub + k)的值为要编码的向量的第m个子向量到第m列子向量的第k个聚类中心的距离  
直接按照列去遍历就可以得到编码结果  

到这里我们就可以分析对多个向量做量化的版本了:
```c++
void ProductQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n) const {
    //分批大小为product_quantizer_compute_codes_bs，默认值为256 * 1024
    //product_quantizer_compute_codes_bs为全局变量，可以更改
    size_t bs = product_quantizer_compute_codes_bs;
    if (n > bs) {
        for (size_t i0 = 0; i0 < n; i0 += bs) {
            size_t i1 = std::min(i0 + bs, n);
            //分批计算量化结果
            compute_codes(x + d * i0, codes + code_size * i0, i1 - i0);
        }
        return;
    }

    if (dsub < 16) { 
        //如果子向量的维数比16小
#pragma omp parallel for
        for (int64_t i = 0; i < n; i++)
            //多线程调用单个向量量化函数就行了
            compute_code(x + i * d, codes + i * code_size);

        } else { 
            //否则，使用BLAS优化将会有不错的效果
            float* dis_tables = new float[n * ksub * M];
            ScopeDeleter<float> del(dis_tables);
            //调用BLAS去计算要编码的向量所有子向量到子向量聚类中心的距离
            compute_distance_tables(n, x, dis_tables);

#pragma omp parallel for
        //按照输入要编码向量的粒度并行
        for (int64_t i = 0; i < n; i++) {
            uint8_t* code = codes + i * code_size;
            const float* tab = dis_tables + i * ksub * M;
            //通过distance_table计算最近的子向量聚类中心
            //每个向量的编码结果放到code中
            compute_code_from_distance_table(tab, code);
        }
    }
}
```

有必要看下compute_distance_tables的实现
```c++
void ProductQuantizer::compute_distance_tables(
        size_t nx,
        const float* x,
        float* dis_tables)  {
    if() {
        //如果dsub比较小就使用朴素的算法，调用avx指令集，使用openMP多线程去做
    } else { 
        //否则使用blas，把向量切分成子向量。调用BLAS接口
        for (int m = 0; m < M; m++) {
            //每次计算一列子向量的距离表
            pairwise_L2sqr(
                    //子向量维数
                    dsub,
                    //所有要进行量化的向量的个数
                    nx,
                    //第m个子向量起始地址
                    x + dsub * m,
                    //子聚类中心个数
                    ksub,
                    //第m列子向量的聚类中心起始地址
                    centroids.data() + m * dsub * ksub,
                    //第m列子向量的距离表的起始地址
                    dis_tables + ksub * m,
                    //完整向量的长度d，向下一个向量步进的长度
                    d,
                    //子向量的长度，子向量聚类中心数组中向下一个聚类中心移动的距离
                    dsub,
                    //一个要编码的向量需要ksub * M个float来存储距离
                    ksub * M);
        }
    }
}
```
pairwise_L2sqr的计算技巧在我们说fvec_L2sqr_ny_y_transposed_ref中已经说过了。
```c++
//计算xb与xq的L2距离的平方
//ldd = ksub * M ---> dis矩阵中到下一个向量要步进的距离，dis矩阵的按照要编码向量的顺序存储
//ldb = dsub ---> xb 矩阵到下一个子向量要步进的距离
void pairwise_L2sqr(...) {
    float* b_norms = dis;
//先计算xb L2范数的平方，放到dis中开头的第一个子向量占用的距离数组中
#pragma omp parallel for if (nb > 1)
    for (int64_t i = 0; i < nb; i++)
        b_norms[i] = fvec_norm_L2sqr(xb + i * ldb, d);
//再计算xq的L2范数的平方
#pragma omp parallel for
    //由于第一个向量被占用，所以先从第二个子向量开始计算L2范数的平方
    for (int64_t i = 1; i < nq; i++) {
        float q_norm = fvec_norm_L2sqr(xq + i * ldq, d);
        //计算 编码子向量与子向量聚类中心的 L2范数平方之和
        for (int64_t j = 0; j < nb; j++)
            dis[i * ldd + j] = q_norm + b_norms[j];
    }
    // 处理第一个子向量的所占用的dis
    {
        float q_norm = fvec_norm_L2sqr(xq, d);
        for (int64_t j = 0; j < nb; j++)
            dis[j] += q_norm;
    }

    {
    FINTEGER nbi = nb, nqi = nq, di = d, ldqi = ldq, ldbi = ldb,lddi = ldd;
    float one = 1.0, minus_2 = -2.0;
    //现在dis中的值为 编码子向量与子向量聚类中心的 L2范数平方之和
    sgemm_("Transposed",
           "Not transposed",
           &nbi,
           &nqi,
           &di,
           &minus_2,
           xb,
           &ldbi,
           xq,
           &ldqi,
           &one,
           dis,
           &lddi);
    }
}
```
最后的sgemm_这个blas调用计算的是如下内容
$$
one \cdot \vec{dis} + minus\_2 \cdot \vec{xb} \cdot \vec{xq}\\ =\vec{dis} - 2 \cdot \vec{xb} \cdot \vec{xq}\\  = ||\vec{xb}||^2 +||\vec{xq}||^2 - 2 \cdot \vec{xb} \cdot \vec{xq}
$$
这个表达式与fvec_L2sqr_ny_y_transposed_ref中分析的完全一致，使用这种方法的好处是可以用BLAS接口快速计算出上面的结果  
这里还有个小技巧，为了减少内存的分配，使用dis中存放第一个向量所占的距离数据的空间来存放xb的L2范数平方。  
最后的结果存放在dis中，dis中的顺序[nq, M, ksub]，nq是要编码的向量的数量，M为划分子向量的数量，ksub为某一子向量上所有聚类中心的数量。
ProductQuantizer还提供了计算内积距离的函数  
计算单个向量的到聚类中心的内积距离:
```c++
void ProductQuantizer::compute_inner_prod_table(
        const float* x,
        float* dis_table) const {
    size_t m;
    //按子向量的顺序挨个计算就行了
    for (m = 0; m < M; m++) {
        fvec_inner_products_ny(
                dis_table + m * ksub,
                x + m * dsub,
                get_centroids(m, 0),
                dsub,
                ksub);
    }
}
```
计算多个向量到聚类中心的距离
```c++
void ProductQuantizer::compute_inner_prod_tables(
        size_t nx,
        const float* x,
        float* dis_tables);
```
该函数的实现和compute_distance_tables几乎一致，只是将pairwise_L2sqr这一步替换为sgemm_，使用blas调用直接计算向量的内积。这里就不分析了

#### 3.2.3 解码函数  
ProductQuantizer提供了解码函数，其作用是将量化后的编码转换为向量，代码如下:
```c++
void ProductQuantizer::decode(const uint8_t* code, float* x) const {
    switch (nbits) {
        case 8:
            faiss::decode<PQDecoder8>(*this, code, x);
            break;

        case 16:
            faiss::decode<PQDecoder16>(*this, code, x);
            break;

        default:
            faiss::decode<PQDecoderGeneric>(*this, code, x);
            break;
    }
}
```
和编码函数类似，解码函数由模板函数decode实现:
```c++
template <class PQDecoder>
void decode(const ProductQuantizer& pq, const uint8_t* code, float* x) {
    PQDecoder decoder(code, pq.nbits);
    for (size_t m = 0; m < pq.M; m++) {
        //获得当前列对应的聚类中心编号c
        uint64_t c = decoder.decode();
        //将聚类中心向量放入code中
        memcpy(x + m * pq.dsub,
               pq.get_centroids(m, c),
               sizeof(float) * pq.dsub);
    }
}
```
PQDecoderGeneric::decode()就是上文encode()的逆过程，简而言之就是从code数组中取出nbits位的数据转换位uint64_t返回，这里不再详细描述。

## 4.MultiIndexQuantizer

这里要提一下MultiIndexQuantizer, 这个类在后续的IndexIVFPQ类中有使用我们先看下这个这个类的声明

```c++
struct MultiIndexQuantizer : Index {
    //上面提到的乘积量化器
    ProductQuantizer pq;
    MultiIndexQuantizer(
            int d,         ///< dimension of the input vectors
            size_t M,      ///< number of subquantizers
            size_t nbits); ///< number of bit per subvector index
    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// add and reset will crash at runtime
    void add(idx_t n, const float* x) override;
    void reset() override;

    MultiIndexQuantizer() {}

    void reconstruct(idx_t key, float* recons) const override;
};
```

可以看到MultiIndexQuantizer是一个Index，自然可以训练和search，但是不支持add和reset。这个类的作用是记录所有PQ量化后的子向量的聚类中心，其search接口实现了将批量输入的物料向量进行PQ量化的功能

### 4.1 训练接口

```c++
void MultiIndexQuantizer::train(idx_t n, const float* x) {
    pq.verbose = verbose;
    pq.train(n, x);
    is_trained = true;
    // 计算有多少个虚拟的物料向量
    ntotal = 1;
    for (int m = 0; m < pq.M; m++)
        ntotal *= pq.ksub;
}
```

本质上就是对ProductQuantizer进行训练。

上面提到MultiIndexQuantizer不支持add和reset，那我们要搜索的物料向量是什么呢？实际上我们并没有存储物料向量，物料向量是虚拟的，其取值范围是PQ后每一维子向量对应的所有聚类中心所有可能组合，即M个子量化器(M个维度)上的聚类中心形成的笛卡尔积。可以想到虚拟物料向量的个数为$$ksub^M$$(M个维度，每个维度有ksub个取值可能)个，这也就是上面的``ntotal``的计算逻辑。这里也很容易明白为什么MultiIndexQuantizer不支持add和reset方法了。

### 4.2 搜索接口

```c++
//按照multi_index_quantizer_search_bs的数量为一批进行搜索
int multi_index_quantizer_search_bs = 32768;
void MultiIndexQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    //分批搜索的逻辑略过
    ...
    float* dis_tables = new float[n * pq.ksub * pq.M];
    ScopeDeleter<float> del(dis_tables);
    //计算距离表(query子向量到子量化器聚类中心的距离表)
    pq.compute_distance_tables(n, x, dis_tables);
        if (k == 1) {
        // 如果我们只要找到最近虚拟物料向量
//按照query向量粒度并行
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            //对于一个query向量，跳转到距离表的位置
            const float* dis_table = dis_tables + i * pq.ksub * pq.M;
            float dis = 0;
            idx_t label = 0;
			//只需要每个子维度上的距离最小的聚类中心即可
            for (int s = 0; s < pq.M; s++) {
                float vmin = HUGE_VALF;
                idx_t lmin = -1;
                for (idx_t j = 0; j < pq.ksub; j++) {
                    if (dis_table[j] < vmin) {
                        vmin = dis_table[j];
                        lmin = j;
                    }
                }
                dis += vmin;
                //记录当前子维度的最小距离对应的聚类中心编号
                label |= lmin << (s * pq.nbits);
                //移动距离表指针到下一个query向量的位置
                dis_table += pq.ksub;
            }
            distances[i] = dis;
            labels[i] = label;
        }
    } else {
#pragma omp parallel if (n > 1)
        {
            //在k>1的情况下，使用MinSumK这个泛型类计算topK问题
            MinSumK<float, SemiSortedArray<float>, false> msk(
                    k, pq.M, pq.nbits, pq.ksub);
//这里的并行粒度也是query向量的等级
#pragma omp for
            for (int i = 0; i < n; i++) {
                msk.run(dis_tables + i * pq.ksub * pq.M,
                        pq.ksub,
                        distances + i * k,
                        labels + i * k);
            }
        }
    }
}
```

这部分的代码结构也比较的清晰，如果我们只需要计算距离最近的物料向量，只需要遍历距离表，找到每个聚类中心的距离最小的子向量即可。对于k>0的情况，使用了MinSumK这个模板类来计算M*ksub的距离表中组合可能的最小的k个值。这里不详细介绍MinSumK这个类，大致介绍下计算的流程

step1.初始化，``sxx[M][ksub]`` 为排序后的距离表(该距离表只排了前k个元素)。计算M维对应的最小距离的和sum，并将每个维度的第1个距离与第0个距离的差值及其对应的编号(1~M)插入堆中

step2. pop出堆中最小的元素，将其编号与当前最小的组合编码合并，得到搜索的结果

step3. 计算M维中每一维当前距离与上一个push入堆的距离的差值diff，将diff push到堆中

step4. 重复step2、step3，直到取到最小的k个组合编码，返回结果

这里的sxx在源码中用的是``std::vector<SemiSortedArray<float>> ssx``来存储的，``SemiSortedArray``为只排序最小k个的数组。
