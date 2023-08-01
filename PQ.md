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
        
        Train_hypercube,    
        Train_hypercube_pca
    };
}
```

