# Leetcode学习笔记
## Leetcode 3005 最大频率元素计数(&哈希表)
涉及到对某些重复元素的计数问题，很自然的想到用哈希。
由于是计数而不是判断是否存在，所以用unordered_map而不是unordered_set。
unordered_set位于头文件\<unordered_set>中。
哈希表在冲突不是很大的情况下查询可以做到O(1)的时间复杂度。通常来说，默认可以被作为哈希的数据类型有以下几种：
整数类、浮点数类、字符串类、指针类(比较的是指针地址本身而非内容)
如果要对其他非标准类型的元素使用哈希表，则需要满足**两个条件**:
1. 必须提供哈希函数(默认为std::hash\<Key>)
2. 必须提供相等比较函数(默认为std::equal_to\<key>)
对于某些自定义类型，你必须提供这两个函数才能使得其可用:
```cpp
struct Point { int x, y; };

struct PointHash {
    size_t operator()(const Point &p) const {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
};

struct PointEqual {
    bool operator()(const Point &a, const Point &b) const {
        return a.x == b.x && a.y == b.y;
    }
};

std::unordered_set<Point, PointHash, PointEqual> s;

```
但如果只是想判断数组类型是否存在，可以使用std::vector或者std::array，只需要自己定义hash函数，通常利用以下方法进行定义：
```cpp
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        size_t seed = 0;
        for (int x : v)
            seed ^= std::hash<int>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            //本质上就是boost::hash_combine(seed, x);
            //该文件位于<boost/functional/hash.hpp>中
        return h;
    }
};
```

具体来说它们的接口主要如下：
Unordered_set:
```cpp
//构造函数
unordered_set();                              // 默认构造
unordered_set(size_type bucket_count);        // 指定桶数
unordered_set(size_type bucket_count, const Hash& hash); // 指定哈希函数
unordered_set(InputIt first, InputIt last);   // 通过范围构造，例如从某个vector的begin()和end()构造
unordered_set(const unordered_set& other);    // 拷贝构造
unordered_set(unordered_set&& other);         // 移动构造
unordered_set<int> a = {1,2,3,4,5,1}; // 利用列表初始化
vector<int> v;
unordered_set<int> a(v.begin(),v.end());
//插入元素
uset.insert(5);                  // 插入元素
uset.emplace(10);                // 原地构造（效率更高）
uset.emplace_hint(uset.begin(), 15); // 提供 hint 对unordered的两类基本上没用
//删除元素
uset.erase(5);        // 根据值删除
uset.erase(uset.begin()); // 根据迭代器删除
uset.erase(it1, it2);     // 删除迭代器区间
//查找元素
auto it = uset.find(10);   // 返回迭代器，没找到返回 uset.end()
bool exists = uset.count(10); // 返回出现次数（0 或 1）
//大小与容量
uset.size();       // 元素个数
uset.empty();      // 是否为空
uset.max_size();   // 理论最大元素数
//迭代器
uset.begin(); uset.end();     // 正向迭代
uset.cbegin(); uset.cend();   // 常量迭代
uset.rbegin(); uset.rend();   // 反向迭代
//清空容器
uset.clear();  // 删除所有元素
```

unordered_map的用法和set类似，位于头文件<unordered_map>中，但是它储存的是一个类似于pair\<const Key, Value\>的结构，因此对于Key的要求和Set一样。主要接口如下：
```cpp
//构造
std::unordered_map<std::string,int> umap;           // 空 map
std::unordered_map<std::string,int> umap2{{"a",1},{"b",2}}; // 列表初始化
//此外，也可以采用迭代器的方式初始化，但是需要保证迭代器指向的元素满足pair<const Key, Value>的格式，不然就报错
umap["apple"] = 5;         // operator[],不存在就插入,但需要Value有默认构造函数，否则报错
std::cout << umap.at("apple") << std::endl;  // at，访问已存在元素，如果不存在会异常
umap.insert({"banana",3});      // 插入 pair
umap.emplace("orange",7);       // 原地构造
umap.erase("banana");           // 按 key 删除
umap.clear();                   // 清空整个 map
//查找
if (umap.find("apple") != umap.end())  // find 返回迭代器
    std::cout << "found apple" << std::endl;

if (umap.contains("orange"))           // C++20 其实也可以用类似uset的count()函数
    std::cout << "contains orange" << std::endl;
//容量
std::cout << "size: " << umap.size() << std::endl;
if (umap.empty()) std::cout << "map is empty" << std::endl;
```

以上就是关于哈希表的主要用法总结。