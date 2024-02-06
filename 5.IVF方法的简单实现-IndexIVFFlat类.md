# Faiss源码阅读
## 1. IndexIVFFlat类
该类IndexIVF类的一个朴素实现，重要的搜索部分并没有重载，添加物料向量的部分也几乎相同，接下来分析下其实现有什么不同:
添加物料向量的接口
```c++
void IndexIVFFlat::add_core(
        idx_t n,
        const float* x,
        const int64_t* xids,
        const int64_t* coarse_idx) {
    int64_t n_add = 0;
    DirectMapAdd dm_adder(direct_map, n, xids);
#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset =
                        invlists->add_entry(list_no, id, (const uint8_t*)xi);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }
    ntotal += n;
}
```
不能说相似，只能说一模一样，不同的点在于
1. 没有分批add物料
2. 不对向量进行encoding

向量编码函数:
```c++
void IndexIVFFlat::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            // 编码后的向量长度为(code_size + coarse_size)
            // 找到当前向量的起始位置
            uint8_t* code = codes + i * (code_size + coarse_size);
            // 取得当前向量的起始地址
            const float* xi = x + i * d;
            if (list_no >= 0) {
                // 对list_no编码，并放到code的起始位置上
                encode_listno(list_no, code);
                // 在list_no编码后放置原始向量
                memcpy(code + coarse_size, xi, code_size);
            } else {
                // list_no不合法，对应向量的位置清零
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}
```

解码函数:
```c++
void IndexIVFFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        // 默认每次取code_size + coarse_size 字节的数据
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        // list_no部分去掉，只保留物料向量部分
        memcpy(xi, code + coarse_size, code_size);
    }
}
```


重建向量的方法:
```c++
void IndexIVFFlat::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}
```
实现很简单，因为倒排表里存储的就是原始向量，所以不需要做任何处理，直接memcpy就完事了

此外还有辅助类IVFFlatScanner，该类重载了distance_to_code方法:
```c++
    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        float dis = metric == METRIC_INNER_PRODUCT
                ? fvec_inner_product(xi, yj, d)
                : fvec_L2sqr(xi, yj, d);
        return dis;
    }
```
fvec_inner_product、fvec_L2sqr已经在前面介绍过，不在赘述。

## 2. IndexIVFFlatDedup
这个类是IndexIVFFlat的子类，在删除、添加、训练物料向量时，检查向量是否相同，若相同则仅记录id。这样可以减少内存使用  

训练函数如下:
```c++
void IndexIVFFlatDedup::train(idx_t n, const float* x) {
    // hash值到id的映射
    std::unordered_map<uint64_t, idx_t> map;
    std::unique_ptr<float[]> x2(new float[n * d]);

    int64_t n2 = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t hash = hash_bytes((uint8_t*)(x + i * d), code_size);
        // hash值相同
        if (map.count(hash) &&
            //检测当前的向量与已经在x2中的向量是否完全相同
            !memcmp(x2.get() + map[hash] * d, x + i * d, code_size)) {
            // 相同，跳过
        } else {
            // 不相同，添加到x2中
            map[hash] = n2;
            memcpy(x2.get() + n2 * d, x + i * d, code_size);
            n2++;
        }
    }
    IndexIVFFlat::train(n2, x2.get());
}
```

添加接口
```c++
void IndexIVFFlatDedup::add_with_ids(
        idx_t na,
        const float* x,
        const idx_t* xids) {
        ...
            for (int64_t o = 0; o < n; o++) {
                // 与已有的向量相同?
                if (!memcmp(codes.get() + o * code_size, xi, code_size)) {
                    offset = o;
                    break;
                }
            }

            if (offset == -1) { //不相同，直接插入
                invlists->add_entry(list_no, id, (const uint8_t*)xi);
            } else {
                // 相同
                idx_t id2 = invlists->get_single_id(list_no, offset);
                std::pair<idx_t, idx_t> pair(id2, id);

#pragma omp critical
                // 记录到instances中
                instances.insert(pair);

            }
        }
```

删除指定id的接口:
```c++
size_t IndexIVFFlatDedup::remove_ids(const IDSelector& sel) {
    std::unordered_map<idx_t, idx_t> replace;
    std::vector<std::pair<idx_t, idx_t>> toadd;
    for (auto it = instances.begin(); it != instances.end();) {
        if (sel.is_member(it->first)) {
            // 第一个id要删除、第二个id不删除的情况
            if (!sel.is_member(it->second)) {
                //把第二个id记录到toadd中
                if (replace.count(it->first) == 0) {
                    replace[it->first] = it->second;
                } else {
                // 1,2相同，2，3相同； 要删除1，那把2，3放在一起作为新的pair记录到toadd中
                    std::pair<idx_t, idx_t> new_entry(
                            replace[it->first], it->second);
                    toadd.push_back(new_entry);
                }
            }
            // 从instances中删除
            it = instances.erase(it);
        } else {
            //第二个id要删除，第一个不删除
            if (sel.is_member(it->second)) {
                //直接删除就行了
                it = instances.erase(it);
            } else {
                ++it;
            }
        }
    }
    //先删除，在把需要添加的加回来
    instances.insert(toadd.begin(), toadd.end());
    //下面执行IndexFlat的删除
    ...
}
```

搜索接口:
```c++
void IndexIVFFlatDedup::search_preassigned() {
    // 先搜索
    IndexIVFFlat::search_preassigned(
        n, x, k, assign, centroid_dis, distances, labels, false, params);
    // 从搜索结果中找到重复的id加到最终的搜索结果中
    for (int64_t i = 0; i < n; i++) {
        idx_t* labels1 = labels + i * k;
        float* dis1 = distances + i * k;
        int64_t j = 0;
        for (; j < k; j++) {
            if (instances.find(labels1[j]) != instances.end()) {
                break;
            }
        }
        if (j < k) {
            int64_t j0 = j;
            int64_t rp = j;
            while (j < k) {
                auto range = instances.equal_range(labels1[rp]);
                float dis = dis1[rp];
                labels2[j] = labels1[rp];
                dis2[j] = dis;
                j++;
                for (auto it = range.first; j < k && it != range.second; ++it) {
                    labels2[j] = it->second;
                    dis2[j] = dis;
                    j++;
                }
                rp++;
            }
            // 有重复，把重复的id添加到结果中
            memcpy(labels1 + j0,
                   labels2.data() + j0,
                   sizeof(labels1[0]) * (k - j0));
            memcpy(dis1 + j0, dis2.data() + j0, sizeof(dis2[0]) * (k - j0));
        }
    }
}
```