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