# Fassi源码简介
## 1. Index类详解
如果你看了第0节中tutorial中的代码，你会发现整个入库、检索的过程都是围绕着Index展开的，是的，``IndexFlatL2``、``IndexIVFPQ``、``IndexIVFFlat``都是Index类的子类。接下来我们就来看看Index类
### 1.1 Index类概述
Index类是一个抽象类，它支持向索引中添加向量以及搜索这些向量。在Index类中，所有添加或搜索时提供的向量都是由32位浮点数组表示的。虽然内部表示可能会有所不同，但用户可以将32位浮点数组作为输入。  

### 1.2 Index成员变量
好吧上面这段注释像是什么都没说，我们先看下Index类的成员变量：
```c++
//fassi/Index.cpp
struct Index {
    //向量的维数
    int d;        ///< vector dimension
    //所有已经被索引的向量的个数
    idx_t ntotal; ///< total nb of indexed vectors
    //当前的索引有没有被train过
    bool is_trained;
    //计算向量距离时用哪种度量？余弦距离、欧式距离等
    MetricType metric_type;
}
```
成员变量的含义是很清晰的，如果我要实现一个IVFQP的向量搜索引擎，我也会添加这些成员变量，注意一点，Index中并没有保存向量的数据结构，需要我们在子类中自己保存  
另外还要注意的一点是，这里要获得向量的维数d，Index类中的函数要传递多个向量时都是用一个指针``const float* x``和向量个数``idx_t n``，那么把x看做一个数组，那数组的长度应该为n*d


### 1.3 Index类成员函数
这部分需要好好梳理一下
```c++
    /** Perform training on a representative set of vectors
     *
     * @param n      nb of training vectors
     * @param x      training vecors, size n * d
     */
    virtual void train(idx_t n, const float* x);
```
n:要作为样本训练的向量的个数  
为什么要有train的过程呢？注释中还说了要传递具有代表性的向量。  
如果你知道IVF算法的话，就知道train的过程是在找聚类中心或者是类似的操作。简单说一下IVF，对于所有的物料向量，我们将其分桶，具体分到哪个桶就是先对物料向量做聚类，看物料向量属于哪一个聚类中心。在搜索时，只去检索属于同一个桶内的向量，这样，以精度损失，换取了搜索速度的提升。IVFQP的思想也是类似的，不过trian的过程更加复杂。  

```c++
    /** Add n vectors of dimension d to the index.
     *
     * Vectors are implicitly assigned labels ntotal .. ntotal + n - 1
     * This function slices the input vectors in chunks smaller than
     * blocksize_add and calls add_core.
     * @param x      input matrix, size n * d
     */
    virtual void add(idx_t n, const float* x) = 0;
```
添加n个vector到index中  
这里解释了第0节的问题，索引的labels是按照添加的顺序来的。这里还提到这个函数会将输入的向量进行分片，分片大小小于blocksize_add，最后执行add_core，看来实际执行入库操作的是add_core


```c++
    /** Same as add, but stores xids instead of sequential ids.
     *
     * The default implementation fails with an assertion, as it is
     * not supported by all indexes.
     *
     * @param xids if non-null, ids to store for the vectors (size n)
     */
    virtual void add_with_ids(idx_t n, const float* x, const idx_t* xids)
```
和上面的add基本类似，但是原先使用的索引labels是``ntotal``的自增label，这里使用的是调用方传入的``xids``。id、label是唯一标识符的不同叫法，当拿到id后，我们就可以去类似meta表中拿到搜索结果的详细信息

```c++
    /** query n vectors of dimension d to the index.
     *
     * return at most k vectors. If there are not enough results for a
     * query, the result array is padded with -1s.
     *
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     * @param distances   output pairwise distances, size n*k
     */
    virtual void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const = 0;
```
检索接口，这个接口看参数的命名就很清晰了  
n:要检索的query向量的个数  
x:query向量  
k:每个query向量要查询k个近邻向量，不够用全为-1的向量补全   
distances:出参，每个近邻向量到query向量的距离  
labels:出参，每个近邻向量的labels  
SearchParameters:搜索参数，后续详细解释

```c++
    /** query n vectors of dimension d to the index.
     *
     * return all vectors with distance < radius. Note that many
     * indexes do not implement the range_search (only the k-NN search
     * is mandatory).
     *
     * @param x           input vectors to search, size n * d
     * @param radius      search radius
     * @param result      result table
     */
    virtual void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const;
```
范围检索接口，查询所有与query向量的距离小于``radius``的向量

```c++
    /** return the indexes of the k vectors closest to the query x.
     *
     * This function is identical as search but only return labels of neighbors.
     * @param x           input vectors to search, size n * d
     * @param labels      output labels of the NNs, size n*k
     */
    virtual void assign(idx_t n, const float* x, idx_t* labels, idx_t k = 1)
            const;
```
与search接口类似，但是只返回索引labels

```c++
    /// removes all elements from the database.
    virtual void reset() = 0;

    /** removes IDs from the index. Not supported by all
     * indexes. Returns the number of elements removed.
     */
    virtual size_t remove_ids(const IDSelector& sel);
```
删除接口，``reset()``删除所有物料向量，``remove_ids(const IDSelector& sel)``:删除指定ids(就是labels)对应物料向量

```c++
    /** Reconstruct a stored vector (or an approximation if lossy coding)
     *
     * this function may not be defined for some indexes
     * @param key         id of the vector to reconstruct
     * @param recons      reconstucted vector (size d)
     */
    virtual void reconstruct(idx_t key, float* recons) const;
```
重建已经存储的的id为key的向量，什么叫重建？在IVFPQ中，物料向量被编码以提高查询速度及节省内存。如果我们需要原始向量那就需要重建的操作。需要注意的是
在IVFPQ中是无法精确的重建原始向量的。
```c++
    /** Reconstruct several stored vectors (or an approximation if lossy coding)
     *
     * this function may not be defined for some indexes
     * @param n        number of vectors to reconstruct
     * @param keys        ids of the vectors to reconstruct (size n)
     * @param recons      reconstucted vector (size n * d)
     */
    virtual void reconstruct_batch(idx_t n, const idx_t* keys, float* recons)
            const;
```
批重建，实际上就是在OpenMP中调用reconstruct()

```c++
    /** Reconstruct vectors i0 to i0 + ni - 1
     *
     * this function may not be defined for some indexes
     * @param recons      reconstucted vector (size ni * d)
     */
    virtual void reconstruct_n(idx_t i0, idx_t ni, float* recons) const;
```
批重建：重建id从i0到i0+ni-1的物料向量，使用OpenMP

```c++
    /** Similar to search, but also reconstructs the stored vectors (or an
     * approximation in the case of lossy coding) for the search results.
     *
     * If there are not enough results for a query, the resulting arrays
     * is padded with -1s.
     *
     * @param recons      reconstructed vectors size (n, k, d)
     **/
    virtual void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const;

```
先检索，再对召回的向量进行重建

```c++
    /** Computes a residual vector after indexing encoding.
     *
     * The residual vector is the difference between a vector and the
     * reconstruction that can be decoded from its representation in
     * the index. The residual can be used for multiple-stage indexing
     * methods, like IndexIVF's methods.
     *
     * @param x           input vector, size d
     * @param residual    output residual vector, size d
     * @param key         encoded index, as returned by search and assign
     */
    virtual void compute_residual(const float* x, float* residual, idx_t key)
            const;
```


