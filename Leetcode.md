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

关于这道题，思路很简单，就是利用umap统计每个独特元素的出现次数后，遍历一遍umap，动态更新max，由于max更新时前面所有的元素出现次数必然小于max，因此可以将已有的计数清零，实现一次迭代完成任务。

## Leetcode 165 比较版本号 && C++ string常用接口
这道题是让我们对1.001、1.0.0.0这种类型的版本号进行比较，其实坑还挺多。
首先是C++的分隔字符串的问题，我一开始以为类似001、000这种数据需要单独处理，但其实C++的stoi可以直接把这些转成正常的整数。
此外，C++没有类似python的split函数这种方便的东西，所以我们只能暴力读取，一个个解析了。
在解析的途中还有一个坑：我是以.作为分隔符分割的，但是最后一个字符分割的时候它也不会碰到.，这样就会直接结束，所以还得给最后一个加特判，但同时要保证最后一个字符会push_back到结果里。我的读取大概是这么写的：
```cpp
for(int i=0;i<version1.size();i++) {
        char ch = version1[i];
        if(ch == '.' || i == version1.size()-1)
        {
            if(i==version1.size()-1){
            num.push_back(ch);
            }
            int t_num = stoi(num);
            num.clear();
            num1.push_back(t_num);
        }
        else
        {
            num.push_back(ch);
        } 
    }
```
用Python最简单:
```python
nums1 = [int(x) for x in version1.split('.')]
```
推荐的方法除了用python的split之外，还可以用双指针，在每一个.之前记录两个字符串读取到的位置，这样就不需要存储每个位置的数字了。

string在C++中常用的接口和函数小结：
string是C++风格字符串的类型，位于头文件\<string\>中，namespace为std。
```cpp
// 构造函数
    std::string s1;                 // 空字符串
    std::string s2("hello");        // 用 C 风格字符串初始化
    std::string s3(s2);             // 拷贝构造
    std::string s4(s2, 1, 3);       // 从 s2 的索引 1 开始，取 3 个字符
    std::string s5(5, 'x');         // 重复字符构造 "xxxxx"
    //或直接赋值
    string s = "abcde";
//大小和访问
    size() length()
//判断是否为空
    empty()
    s[n] / s.at(n)	//访问第 n 个字符，at 会做边界检查
    front() / back()	//返回第一个/最后一个字符
    c_str()	//返回 C 风格字符串（const char*）
//修改
    s.clear()	//清空字符串
    s.push_back(c)	//在末尾添加字符
    s.append(str)	//在末尾追加字符串
    s.insert(pos, str)	//在 pos 位置插入字符串
    s.replace(pos, len, str)	//替换指定位置和长度的子串
    s.erase(pos, len)	//删除子串
    s.pop_back()	//删除最后一个字符
    s.resize(n)	//调整长度，多出来用默认填充
//查找
    s.find(sub)	//查找子串第一次出现的位置
    s.rfind(sub)	//查找子串最后一次出现的位置
    s.find_first_of(chars)	//查找任意一个字符第一次出现的位置
    s.find_last_of(chars)	//查找任意一个字符最后一次出现的位置
    s.find_first_not_of(chars)	//查找第一个不在 chars 中的字符
    s.find_last_not_of(chars)	//查找最后一个不在 chars 中的字符
    //返回 std::string::npos 表示未找到
//比较
    s1 == s2 / != / < / >	//运算符比较
    s1.compare(s2)	//返回 <0, 0, >0 分别表示 s1 小于、等于、大于 s2
//子串
    std::string sub = s.substr(pos, len); // 从 pos 开始取 len 个字符
//转换
    // string -> int / double
    int x = stoi("123");
    long long y = stoll("123456789");
    double d = stod("3.14");

    // int / double -> string
    std::string s = std::to_string(123);
//遍历，也可以采用begin() end()的迭代器
    for (auto it = s.begin(); it != s.end(); ++it) cout << *it;
```
