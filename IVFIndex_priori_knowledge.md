# Fassi源码阅读
## 1. IVF方法简述
为了提高检索速度，我们可以牺牲一些准确度，只检索部分向量。我们希望选取离query向量近的一些向量。于是一个简单的想法是先对所有物料向量做聚类，得到N个聚类中心，在搜索时，先计算query向量与聚类中心的距离，找到最近的nlist个聚类中心，只与这nlist个聚类中心所代表的向量计算距离，找到其中最近的k个。这就是基于倒排表的向量检索方法。  
这里的倒排表只是借用了传统的全文检索的搜索引擎中的词，可以把聚类中心理解为lucene中的倒排拉链的term，把属于该聚类中心的向量理解为倒排拉链中的doc

## 2. 辅助类介绍
### 2.1 Level1Quantizer类
level1指对物料向量只做一次量化。
先考察Level1Quantizer类的成员变量
```c++
struct Level1Quantizer {
    /// quantizer that maps vectors to inverted lists
    // 把向量映射为倒排表的量化器
    Index* quantizer = nullptr;
    // 倒排表的数量
    size_t nlist = 0;

    /**
     * = 0: use the quantizer as index in a kmeans training
     * 等于0时，quantizer只是单纯的作为k-means聚类时的索引(调用quantizer的search接口找最近的向量)
     * = 1: just pass on the training set to the train() of the quantizer
     等于1时，把训练集(物料向量)传递给quantizer的train()来进行训练
     * = 2: kmeans training on a flat index + add the centroids to the quantizer
     等于2时，在上文的flatindex上做kmeans聚类，最后把聚类中心点加入到quantizer中
     */
    char quantizer_trains_alone = 0;

    //quantizer属于这个Level1Quantizer吗？属于的话在析构函数中释放
    bool own_fields = false;
    // 聚类时使用的index
    Index* clustering_index = nullptr;
    // 聚类的一些参数
    ClusteringParameters cp;
}
```
这里解答原来的一个疑惑，为何quantizer是Index类型？难道Index也是一种量化器吗？  
如果我们把通过物料向量聚类得到的聚类中心向量加入到Index中，再在该Index上搜索离输入向量最近的向量，用搜索结果代替原始的输入向量，这样我们就可以把输入向量量化到某个聚类中心上，从这个角度上看，把Index作为一种量化器是很自然而合理的事。

接下来考察其成员函数
```c++
    /// Trains the quantizer and calls train_residual to train sub-quantizers
    void train_q1(size_t n,const float* x,bool verbose,
            MetricType metric_type) {
        size_t d = quantizer->d;
        if (quantizer_trains_alone == 1) {
            //单纯训练quantizer
            quantizer->train(n, x);
        } else if (quantizer_trains_alone == 0) {
            //新建一个Clustering，接下来介绍
            Clustering clus(d, nlist, cp);
            quantizer->reset();
            if (clustering_index) {
                //若clustering_index不为空，使用clus聚类
                clus.train(n, x, *clustering_index);
                //然后把训练结果得到的聚类中心加入到quantizer中
                quantizer->add(nlist, clus.centroids.data());
            } else {
                //在Clustering中介绍
                clus.train(n, x, *quantizer);
            }
            quantizer->is_trained = true;
        }  else if (quantizer_trains_alone == 2) {
            Clustering clus(d, nlist, cp);
            if (!clustering_index) {
                //clustering_index为空，使用IndexFlatL2作为clus聚类时的index
                IndexFlatL2 assigner(d);
                clus.train(n, x, assigner);
            } else {
                //否则使用clustering_index聚类
                clus.train(n, x, *clustering_index);
            }
            if (!quantizer->is_trained) {
                //使用上面训练出的聚类中心继续训练quantizer
                quantizer->train(nlist, clus.centroids.data());
            }
            //聚类中心加入quantizer
            quantizer->add(nlist, clus.centroids.data());
        }
    }
```
先看下train_q1的参数：  
n:有多少个要训练的向量  
x:向量数组  
metric_type:L2/内积/.etc  
代码结构很清晰，根据quantizer_trains_alone取值走不同的道路，我们一条一条分支看。  
1. quantizer_trains_alone = 1的情况，quantizer自己训练就行了，具体训练是做什么在不同的算法对应的类中有不同实现，在IVF中我们可以简单的理解为训练就是计算聚类中心就行了

2. quantizer_trains_alone == 0或2时，新建了Clustering类的对象clus，clus中保存了向量维数d、聚类中心的数量nlist，及聚类时要用到的一些参数cp。最后使用clus.train()来实现训练过程。  
train()中会调用train_encoded，我们直接看下关键函数train_encoded
照例先介绍下每个参数:  
nx:要训练的向量个数  
x_in:保存向量的数组  
codec:编/解码器，x_in可能是被编码后的数组，需要codec从数组中解析原始向量  
index:训练时使用的Index
weights:训练向量的权重
```c++
void Clustering::train_encoded(
        idx_t nx,const uint8_t* x_in, const Index* codec,Index& index,
        const float* weights) {
        if (!codec) {
            //没有codec，把x_in当做32位float数组
            const float* x = reinterpret_cast<const float*>(x_in);
        }
        const uint8_t* x = x_in;
        //RAII，保存采样后的指针
        std::unique_ptr<uint8_t[]> del1;
        //RAII, 保存采样后的向量权重
        std::unique_ptr<float[]> del3;
        //检查所有输入向量的数量是不是超过允许的聚类样本的数量
        //max_points_per_centroid来着上面的cp，代表聚类后每个类最多允许有多少个样本，k为聚类中心的数量
        if (nx > k * max_points_per_centroid) {
            uint8_t* x_new;
            float* weights_new;
            //超过了就进行采样，使其参与聚类的样本的数量满足等于 k * max_points_per_centroid
            nx = subsample_training_set(
                    *this, nx, x, line_size, weights, &x_new, &weights_new);
            del1.reset(x_new);
            x = x_new;
            del3.reset(weights_new);
            weights = weights_new;
        } // 样本过少，比k * min_points_per_centroid还小就报warning
        ... 
        //处理corner case
        ...
        // 记录最佳迭代的容器
        std::vector<ClusteringIterationStats> best_iteration_stats;
        std::vector<float> best_centroids;//大小k * d
        for (int redo = 0; redo < nredo; redo++) {
            // 随机采样作为处理的聚类中心，这里也没使用类似kmeans++等技巧
            rand_perm(perm.data(), nx, seed + 1 + redo * 15486557L);
            if (!codec) {
                //不使用codec，默认以32位float类型处理
                for (int i = n_input_centroids; i < k; i++) {
                    memcpy(&centroids[i * d], x + perm[i] * line_size,  line_size);
                }
            } else {
                //使用codec自带的编码器
                for (int i = n_input_centroids; i < k; i++) {
                    codec->sa_decode(1, x + perm[i] * line_size, &centroids[i *     d]);
                }
            }
            //可选的处理cp.spherical控制是否需要对聚类中心做L2归一化操作
            //cp.int_centroids控制是否要将聚类中心取整
            post_process_centroids();

            //处理index，如果index没有train过，使用当前的聚类中心train一次，再把聚类    中心加入到index中
            //为什么要先train一次？提高下面使用index检索的速度
            ...

            //开始kmeans迭代
            for (int i = 0; i < niter; i++) {
                if (!codec) {
                    //不使用codec，直接去检索，获得到聚类中心的距离dis和聚类中的id保    存在assign中
                    index.search(nx, reinterpret_cast<const float*>(x), 1,
                            dis.get(),assign.get());
                } else {
                    //每次处理decode_block_size的vector
                    for() {
                        //先解码
                        codec->sa_decode(
                            i1 - i0, x + code_size * i0, decode_buffer.data());
                        //再搜索距离哪个聚类中心最近
                        index.search(i1 - i0,decode_buffer.data(),1,
                                dis.get() + i0,assign.get() + i0)
                    }
                }

            //计算所有的距离
            obj = 0;
            for (int j = 0; j < nx; j++) {
                obj += dis[j];
            }

            //更新聚类中心，使用OpenMP优化
            compute_centroids(d,k,nx,k_frozen,x,codec,assign.get(),weights,
            hassign.data(),centroids.data());

            //如果出现空的聚类，则找一个包含权重高的聚类中心，略微调整其聚类中心大小，从而把这个类分成两个类
            int nsplit = split_clusters(
            d, k, nx, k_frozen, hassign.data(), centroids.data());

            //记录统计信息到iteration_stats中
            ...

            //后处理，上文介绍过，不赘述
            post_process_centroids();
            // 将新的聚类中心设置到index中
            index.reset();
            if (update_index) {
                index.train(k, centroids.data());
            }

            index.add(k, centroids.data());
            }
            //所有的k-means迭代结束了，比较这次的结果是否比当前的最好的结果要好
            //如果是的话，更新最优的结果
            ...
        }
        //所有的redo做完了，取最优的解
        centroids = best_centroids;
        iteration_stats = best_iteration_stats;
        index.reset();
        index.add(k, best_centroids.data());
        //现在index中保存了最优的k个聚类中心了
    }
```

真长啊！现在我们知道了clus.train()就是把输入的向量做聚类然后把聚类中心放到train()函数传入的index里，其实quantizer_trains_alone的取值就是决定了在训练时要不要把quantizer传入Clustering中，如果不传入的话就使用外部的clustering_index，如果clustering_index为空指针的话，就新建一个IndexFlatL2来代替clustering_index，在聚类结束后，把聚类中心的点放入quantizer中。

在Level1Quantizer中还有几个辅助方法，都是为listno(倒排拉链的编号)进行压缩:
```c++
//listno在压缩后占多少个字节
size_t Level1Quantizer::coarse_code_size() const;
//压缩list_no到code中
void Level1Quantizer::encode_listno(idx_t list_no, uint8_t* code) const;
// 从code中还原处理list_no
idx_t Level1Quantizer::decode_listno(const uint8_t* code) const;
```

### 2.2 InvertedLists接口及其实现
InvertedLists规定了倒排表的接口，其实现类要支持多线程读、写，对于更改InvertedLists中的倒排拉链大小、添加新的倒排拉链，只需要保证在处理不同的倒排拉链时线程安全即可
```c++
struct InvertedLists {
    //有多少个倒排拉链
    size_t nlist;     ///< number of possible key values
    //每个向量占用多少个字节的大小
    size_t code_size; ///< code size per vector in bytes
    // 遍历当前的倒排表能使用迭代器吗
    bool use_iterator;
}
```
读取方法
``` c++
struct InvertedLists {
    // 检查list_no对应的倒排拉链是否为空
    bool is_empty(size_t list_no) const;

    // list_no对应的倒排拉链中有多少个向量
    virtual size_t list_size(size_t list_no) const = 0;
    // 获得list_no对应的uint8数组，该数组是对向量集合的编码
    virtual const uint8_t* get_codes(size_t list_no) const = 0;
    // 获得list_no对应的id数组(也就是前文提到的labels)
    virtual const idx_t* get_ids(size_t list_no) const = 0;
    // 释放codes资源，delete codes
    virtual void release_codes(size_t list_no, const uint8_t* codes) const;
    // 释放ids资源，delete ids
    virtual void release_ids(size_t list_no, const idx_t* ids) const;
    // 获得list_no对应的倒排拉链，从倒排拉链中获得offset位置上的id
    virtual idx_t get_single_id(size_t list_no, size_t offset) const;
    // 获得list_no对应的倒排拉链，从倒排拉链中获得offset位置上的向量
    virtual const uint8_t* get_single_code(size_t list_no, size_t offset) const;
    // 预取指定list_no的倒排表,eg.在OnDiskInvertedList中，会通过在该方法中启动多个线程从磁盘上预取倒排表
    void InvertedLists::prefetch_lists(const idx_t*, int) const;
    // 获得list_no对应倒排拉链的迭代器
    virtual InvertedListsIterator* get_iterator(size_t list_no) const;
}
```
写入方法
```c++
    // 在list_no对应的倒排拉链中，添加物料对应的向量与id
    // 多提一句，感觉这里应该是把物料对应的<id,vector>称为一个entry，下面我也用entry来代替<id,vector>
    virtual size_t add_entry(size_t list_no, idx_t theid, 
const uint8_t* code);
    // 添加多个entry，在ArrayInvertedLists的实现中是添加在末尾
    virtual size_t add_entries(size_t list_no,size_t n_entry,
    const idx_t* ids,const uint8_t* code) = 0;
    // 在list_no对应的倒排拉链，将offset位置上的id、vector替换为新的id、code
    virtual void update_entry(size_t list_no,size_t offset,
    idx_t id,const uint8_t* code);
    // 在list_no对应的倒排拉链上，将offset位置后n_entry长度上的entries用<ids,code>替代
    virtual void update_entries(size_t list_no,size_t offset,size_t n_entry,
            const idx_t* ids,const uint8_t* code);
    // 更改list_no对应的倒排拉链的长度
    virtual void resize(size_t list_no, size_t new_size) = 0;
    //清空整个倒排表
    virtual void reset();
```

一些高层的接口
```c++
    //将另一个InvertedLists合并到当前lists中，add_id是在添加时oivf中的id数组全部加上add_id这个值
    void merge_from(InvertedLists* oivf, size_t add_id) {
 #pragma omp parallel for
 //  按倒排拉链并行
    for (idx_t i = 0; i < nlist; i++) {
        size_t list_size = oivf->list_size(i);
        ScopedIds ids(oivf, i);
        if (add_id == 0) {
            //添加倒排表oivf中id为i的倒排拉链添加到到当前倒排表中
            //ArrayInvertedLists的实现是添加到当前list_no=i的倒排拉链尾部
            add_entries(i, list_size, ids.get(), ScopedCodes(oivf, i).get());
        } else {
            std::vector<idx_t> new_ids(list_size);
            //先把每个id加上add_id
            for (size_t j = 0; j < list_size; j++) {
                new_ids[j] = ids[j] + add_id;
            }
            // 再添加到倒排拉链中
            add_entries(
                    i, list_size, new_ids.data(), ScopedCodes(oivf, i).get());
        }
        //释放oivf倒排表的资源
        oivf->resize(i, 0);
    }
}
    }
    //将当前的invertedlist中的元素拷贝到other中，具体拷贝那些id参考subset_type_t的注释，这里不赘述
    size_t copy_subset_to(InvertedLists& other,subset_type_t subset_type,
    idx_t a1,idx_t a2) const;
```

一些统计接口
```c++
// 衡量倒排拉链长度平衡程度，=1为完全平衡，> 1 不平衡
double InvertedLists::imbalance_factor();
// 所有倒排拉链长度之和
size_t compute_ntotal() const;
```

所有的接口介绍完了，大家应该也知道该怎么实限这些接口，我们以ArrayInvertedLists为例简单看下实现。  
ArrayInvertedLists是用std::vector存储的在内存中使用的倒排表，这也是默认的实现。
```c++
struct ArrayInvertedLists : InvertedLists {
    // 物料对应的向量放在codes里面，一共有nlist个std::vector<uint8_t>(倒排拉链的向量部分)
    std::vector<std::vector<uint8_t>> codes; // binary codes, size nlist
    // 物料对应的id放在ids里面，一共有nlist个std::vector<idx_t>(倒排拉链的id部分)
    std::vector<std::vector<idx_t>> ids;     ///< Inverted lists for indexes
}
```
重载的方法比较重要的点:
```c++
size_t ArrayInvertedLists::add_entries(
    size_t list_no,size_t n_entry,
    const idx_t* ids_in,const uint8_t* code) {
    if (n_entry == 0)
        return 0;
    assert(list_no < nlist);
    size_t o = ids[list_no].size();
    //先扩容
    ids[list_no].resize(o + n_entry);
    //将要添加的entry添加到vector的尾部
    memcpy(&ids[list_no][o], ids_in, sizeof(ids_in[0]) * n_entry);
    codes[list_no].resize((o + n_entry) * code_size);
    memcpy(&codes[list_no][o * code_size], code, code_size * n_entry);
    return o;
}

void ArrayInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes_in) {
    assert(list_no < nlist);
    //update时，update的内容不能超过原始vector的大小，即update_entries不能自动扩容
    assert(n_entry + offset <= ids[list_no].size());
    memcpy(&ids[list_no][offset], ids_in, sizeof(ids_in[0]) * n_entry);
    memcpy(&codes[list_no][offset * code_size], codes_in, code_size * n_entry);
}
```
add_entries、update_entries的实现很简单，我们只需要注意add_entries是将新的entries加入到尾部，而update_entries是不能自动扩容的。  
还有一些简单接口
```c++
size_t ArrayInvertedLists::list_size(size_t list_no) const {
    assert(list_no < nlist);
    // 直接取ids[list_no].size()作为每个的倒排拉链的长度
    return ids[list_no].size();
}

const uint8_t* ArrayInvertedLists::get_codes(size_t list_no) const {
    assert(list_no < nlist);
    //返回codes这个vector的内部指针
    return codes[list_no].data();
}

const idx_t* ArrayInvertedLists::get_ids(size_t list_no) const {
    assert(list_no < nlist);
    //返回ids这个vector的内部指针
    return ids[list_no].data();
}

void ArrayInvertedLists::resize(size_t list_no, size_t new_size) {
    //同时resize ids、codes
    ids[list_no].resize(new_size);
    codes[list_no].resize(new_size * code_size);
}
```
这里的代码很简单，不用我解释大家也能看懂。

总结下ArrayInvertedLists的特点
1. 不能调整倒排拉链的数量，倒排拉链的数量始终为构造函数的中传入的nlist
2. 添加entries时，是直接添加到ids、codes这两个vector的尾部
3. update_entries时不能自动扩容
4. 仅在内存内使用，没有持久化到磁盘的功能

### 2.3 DirectMap
对于给定的物料的唯一标识符id，我们可以通过DirectMap在O(1)的时间内找到该物料属于的list_no和在该list_no中的offset(偏移)。内部实现为数组实现:``std::vector<idx_t>``或hash表实现``std::unordered_map<idx_t, idx_t>``。  
如果设置类型为NoMap则不存储任何id到entry的映射  
注意hash表的value与vector中的value类型为idx_t，64位整型，高32位为list_no，低32位为id对应的元素在invertedLists中的偏移offset，fassi中把这个64位整形称为lo，并提供了lo_build、lo_listno、lo_offset来操作lo
```c++
    // 设置用数组还是hash表，并使用invlists初始化该DirecMap
    void DirectMap::set_type(Type new_type, const InvertedLists* invlists, size_t ntotal);
    
    // 通过id拿到lo
    idx_t get(idx_t id) const;

    // 如果type为数组，则不能添加<id,lo>数组
    void check_can_add(const idx_t* ids);

    // 添加一个<id,lo>
    // 对于array，直接调用std::vector::push_back()
    // 对于HashTable，直接调用hashtable[id] = lo
    void add_single_id(idx_t id, idx_t list_no, size_t offset);
    // 清空容器
    void clear();

```
显然add_single_id不是线程安全的
特别说一下删除方法:
```c++
size_t DirectMap::remove_ids(const IDSelector& sel, InvertedLists* invlists)；
```
这里需要传入IDSelector、invlists的指针，remove_ids会删除invlists中在id在sel中的元素，如果DirectMap的类型为HashTable，则在DirectMap中同步invlists中的修改，如果为NoMap则什么事也不做，如果type为array则抛出异常。  

更新方法:
```c++
void DirectMap::update_codes(
        InvertedLists* invlists,int n,const idx_t* ids,
        const idx_t* assign,const uint8_t* codes);
```
只有数组实现的DirectMap才支持update_codes方法，其实现是把数组最后一个entry移动到要更新的id对应的offset位置(先删除)，然后把要更新的id插入到InvertedLists对应倒排拉链的尾部，在以上过程中invlists与DirectMap会同步更新，保证DirectMap中id到lo的映射是正确的。

由于DirectMap类的add_single_id方法不是线程安全的，所以我们需要一个辅助类DirectMapAdd，来实现对DirectMap的线程安全的操作。这个类有必要分析一下

```c++
struct DirectMapAdd {
    // 实际要更新的DirectMap
    DirectMap& direct_map;
    // DirectType类型
    DirectMap::Type type;
    // 原始DirectMap的LO的个数
    size_t ntotal;
    // 在原始DirectMap上要扩展n个LO
    size_t n;
    /* 在Type = hashtable的情况下，由于可以添加不连续的id，所以要使用xid记录调用add时添加的i到要添加的真实id的映射 */
    const idx_t* xids;

    // 用于临时缓存添加到hashtable类型的DirectMap的lo
    std::vector<idx_t> all_ofs;

    // 在DirectMap为array类型时，i为物料的id
    // 在DirectMap为Hashtable类型时，物料的id为xid[i]
    void add(size_t i, idx_t list_no, size_t offset);
}
```
构造函数
```c++
DirectMapAdd::DirectMapAdd(DirectMap& direct_map, size_t n, const idx_t* xids)
        : direct_map(direct_map), type(direct_map.type), n(n), xids(xids) {
    if (type == DirectMap::Array) {
        FAISS_THROW_IF_NOT(xids == nullptr);
        ntotal = direct_map.array.size();
        direct_map.array.resize(ntotal + n, -1);
    } else if (type == DirectMap::Hashtable) {
        // can't parallel update hashtable so use temp array
        all_ofs.resize(n, -1);
    }
}
```
构造函数中预分配了空间，并在ntotal中记录了当前DirectMap存了多少数据  

添加LO的方法
```c++
void DirectMapAdd::add(size_t i, idx_t list_no, size_t ofs) {
    if (type == DirectMap::Array) {
        //如果类型为array，则会直接对array[ntotal + i]赋值
        direct_map.array[ntotal + i] = lo_build(list_no, ofs);
    } else if (type == DirectMap::Hashtable) {
        // 否则记录到all_ofs数组中
        all_ofs[i] = lo_build(list_no, ofs);
    }
}
```
该接口实现了将指定的id对应的LO添加到DirectMap中。需要注意的是，如果DirectMap类型为HashMap，在析构函数中，才把all_ofs数组这个缓存放到direct_map.hashtable中，注意取id时优先取xids中的数据，代码如下所示:
```c++
DirectMapAdd::~DirectMapAdd() {
    if (type == DirectMap::Hashtable) {
        for (int i = 0; i < n; i++) {
            idx_t id = xids ? xids[i] : ntotal + i;
            direct_map.hashtable[id] = all_ofs[i];
        }
    }
}
```


DirectMapAdd总结:  
该类是对DirectMap的包装，虽然注释中写了其提供了线程安全的方法，但实际上我们还是需要在代码中保证，不同的线程不能添加同一个id及其对应的lo信息。


### 2.4 InvertedListScanner
这个类是一个抽象类，用于辅助遍历倒排表
```c++
struct InvertedListScanner {
    // 当前的list_no
    idx_t list_no = -1;    
    // 记录最大还是最小
    bool keep_max = false; 
    // true:存储LO false:存储id
    bool store_pairs;

    /// 遍历时使用的IDSelector
    const IDSelector* sel;

    InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr)
            : store_pairs(store_pairs), sel(sel) {}

    /// 遍历时一个向量占多少字节
    size_t code_size = 0;
    // 设置query向量
    virtual void set_query(const float* query_vector) = 0;

    // 设置当前的遍历的倒排表为编号为list_no的倒排表
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;

    // 计算当前query到code这个向量的距离
    virtual float distance_to_code(const uint8_t* code) const = 0;
}
```
重点介绍下InvertedListScanner的以下几个方法:  
scan_codes方法字面意思是扫描物料向量，但实际上承担了计算query向量到物料向量的距离，在distances、labels数组上对物料向量根据距离进行建堆的任务
```c++
    virtual size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const {
        size_t nup = 0;
        if (!keep_max) {
            //小顶堆
            for (size_t j = 0; j < list_size; j++) {
                // 计算query到物料向量的距离
                float dis = distance_to_code(codes);
                // heapify操作
                // 比堆顶元素小，要更新
                if (dis < simi[0]) {
                    // 根据store_pair在labels中放LO或者物料的id
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    // 调整堆
                    maxheap_replace_top(k, simi, idxi, dis, id);
                    nup++;
                }
                codes += code_size;
            }
        } else {
            //大顶堆的处理
            ...
        }
        return nup;
    }
```
输入参数:
|||
|:---:|:---:|
n|物料向量的个数
codes|物料向量数组，大小n*code_size
ids|物料向量的id
k|堆的大小，保留距query向量最近的k个向量

输出参数:
|||
|:---:|:---:|
labels|物料向量的唯一标识符
distances|堆中label对应的物料向量到query向量的距离

返回值:堆调整的次数

iterate_codes方法与scan_codes完全一致，只是把输入参数包装在InvertedListsIterator中，这里不赘述
```c++
virtual size_t iterate_codes(
        InvertedListsIterator* iterator,
        float* distances,
        idx_t* labels,
        size_t k,
        // 实际从iterator中读取了多少个向量
        size_t& list_size) const;
```

与范围搜索相关的方法:
scan_codes_range方法计算了物料向量到query向量的距离，按照store_pairs将LO或者id放入到RangeQueryResult中
```c++
void InvertedListScanner::scan_codes_range(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float radius,
        RangeQueryResult& res) const {
    for (size_t j = 0; j < list_size; j++) {
        float dis = distance_to_code(codes);
        bool keep = !keep_max
                ? dis < radius
                : dis > radius; // TODO templatize to remove this test
        if (keep) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            res.add(dis, id);
        }
        codes += code_size;
    }
}
```
由于距离小于radius的向量的个数是不确定的，所以干脆让外部提供RangeQueryResult接口，作为接受容器  
实现很简单，这里不解释了，做不做排序什么的看RangeQueryResult调用方的实现
```c++
void InvertedListScanner::iterate_codes_range(
        InvertedListsIterator* it,
        float radius,
        RangeQueryResult& res,
        size_t& list_size) const;
```
范围扫描的方法iterate_codes_range，和scan_codes_range完全一致。


## 3. 总结
辅助类介绍完了，现在可以正式介绍IVFIndex了，不过在介绍了这么多辅助类之后，实现一个自己的IndexIVF应该是很简单的事。
 