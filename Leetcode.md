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


## Leetcode 166 分数到小数  && 长除法

总算是碰到了一道我没什么思路的题目(有了思路做的也磕磕绊绊)，但其实有的时候得出结果比想象的要简单，这道题本质上就是让你手工模拟一个长除法，计算两个整数相除得到的结果以无限循环小数的形式写出来。

第一个坑点：符号溢出
因为整数就有正有负，而有的时候做模运算或者除运算得出负数结果会很难办，所以我们先统一符号。然而当 x = INT_MIN的时候，对其取绝对值abs会导致溢出报错(因为-INT_MIN = INT_MAX+1)
所以我们要先对被除数(numerator)和除数(denominaator)进行强制类型转换为long long,之后的其他结果最好也要这样，例如商(quotient)和余数(remainder)

第二个坑点：到底存什么
其实，存的应该是乘以10倍前的那个余数和它第一次在小数中出现的位置，这样当第二次出现这个数的时候就可以直接在字符串中插入括号了

第三个坑点：老想着先存结果
这道题由于模拟的是长除法，我们都知道长除法本质上就是一步步在输出结果，所以完全可以边输出结果边计算，不需要额外操作，不需要额外空间存储。

第四个坑点：逻辑错误
我中间有一步骤的逻辑太复杂了，导致自己都卡出bug，要不是leetcode给了错误示例估计永远都找不出为什么出bug

第五个坑点：分类讨论太复杂
还是一样的原因，你没有自己手动去算一遍，没想清楚到底该怎么操作。
其实本质上，长除法对于有限小数的步骤和对于能整除的情况并无什么区别，只要某次的余数为0即表示计算结束。

这次比较有进步的一点是对哈希容器、string的特性基本上都已经记忆得比较清楚了，这说明刷题和记录还是有用的，之后还要多复习多反思。写代码效率也确实慢，主要还是脑子不清楚，不知道自己到底应该写一个怎么样的算法，我觉得有的时候应该先动笔动脑再动手，用笔把自己的想法记下来，就可以很容易看出问题所在。

附上GPT给出的AC答案学习：
```cpp
class Solution {
public:
    string fractionToDecimal(int numerator, int denominator) {
        if(numerator == 0) return "0"; // 0 直接返回

        string result;
        // 符号处理
        if ((numerator < 0) ^ (denominator < 0)) result += "-";

        // 转 long long 防止溢出
        long long num = abs((long long)numerator);
        long long den = abs((long long)denominator);

        // 整数部分
        long long integerPart = num / den;
        result += to_string(integerPart);

        long long remainder = num % den;
        if(remainder == 0) return result; // 整除

        result += "."; // 小数点
        unordered_map<long long, int> remPos; // 余数 → 小数位下标
        string decimalPart;

        while(remainder != 0) {
            // 余数重复 → 循环节
            if(remPos.count(remainder)) {
                decimalPart.insert(remPos[remainder], "(");
                decimalPart += ")";
                break;
            }

            remPos[remainder] = decimalPart.size(); // 记录余数出现位置
            remainder *= 10;
            decimalPart += to_string(remainder / den);
            remainder %= den;
        }

        result += decimalPart;
        return result;
    }
};

```

## Leetcode 120 三角形最小路径和 && 简单DP

这是一道极其经典的DP题，题意也很简单，就是让你从一个金字塔去求自顶向下/自底向上的路径和最小值（因为自底向上最后都会经过根节点，所以本质上没有区别）

思路就分为自顶向下和自底向上两种，自底向上的最后统计起来简洁一些并且不需要最后求最小值，所以更推荐。

## Leetcode 611 有效三角形的个数 && 双指针入门

给定一个包含非负整数的数组 nums ，返回其中可以组成三角形三条边的三元组个数。

由于组成三角形的条件是任意两边之和大于第三边，那么固定较小的两边，则只需要较小的两边之和大于第三边即可。将数组排序之后，朴素的暴力思路是将三条边按顺序枚举即可，可以轻易算出结果，但复杂度为O(n^3)。

考虑到以下结论：i\<j\<k，当i固定时，若某组i j k满足nums\[i]+nums\[j] \> nums\[k]，则当j增大时对应的k也必然增大，不用再回退进行寻找。也就是说，i固定时j和k是单调增的，可以在同一次枚举中完成，二者最多枚举2*n次。这样两个变量的变化互相影响，保证单调时，即可采用双指针解决问题。

我们固定i，遍历j，如果找到了某个k使得nums\[i]+nums\[j] \> nums\[k]，则说明从j+1到k-1的所有数都可以作为有效三元组，结果增加k-j-1。由于k和j单调，j增大以后k不可能变小，所以k不需要回退，继续增大j遍历k即可。

最后，总的时间复杂度为$O(n^2)$。

## Leetcode 812 最大三角形面积 && 三角形面积求法 && Andrew算法和凸包、向量的外积运算

题目是让我们求三个点之间组成三角形面积的最大值，给了任意三个点坐标求出面积最大值的公式可以用以下形式表示：
$ S_{\triangle ABC}=\frac{1}{2}|(x_1y_2+x_2y_3+x_3y_1)-(y_1x_2+y_2x_3+y_3x_1)| $
由于这个公式的交替性质，这个公式也被称为鞋带公式。

由于题目的数据量很小，所以完全可以暴力枚举来解决，时间复杂度为$O(n^3)$。

不过，当数据量变大的时候，这种方法就不合适了，然而这个问题要想解决必须求出组成这些点集合的凸包的点集，我们可以证明只有在凸包上的点组成的三角形才是面积最大的，否则固定另外两个点，在土包上必然能找到一个高更高的点。


具体求解需要用到Andrew算法，这个我们明天研究。

好，今天我们彻底解决凸包问题。
Andrew算法又被称为单调链算法，其实就是一种贪心的求凸包算法。简单来说，将平面上的所有点按照x轴坐标大小从小到大排序，若x轴大小一样则按y轴大小从小到大排序，这样排序的点集最小值点和最大值点分别是最左下角和最右上角的点,这两个点是必然在凸包内的，我们把按逆时针顺序从最小点到最大点的凸包成为下凸包，反之成为上凸包，Andrew算法就是分别求这两个凸包的过程。

排序可以用以下方法定义：
```cpp
struct Point {
    int x,y;
    bool operator== const (const Point & p)
    {
        return (x==p.x)&&(y==p.y);
    }
}

vector<Point> points;

sort(points.begin(),points.end(),[](const Point &a, const Point &b){
    return (a.x!=b.x) ? (a.x < b.x) : (a.y < b.y); //sort要求使用<
})


```

向量外积

在三维空间中，两个向量$\mathbf{A}(a_x,a_y,a_z), \mathbf{B}(b_x,b_y,b_z)$的外积被定义为：
$$\mathbf{A} \times \mathbf{B} =
\begin{vmatrix} 
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
a_x & a_y & a_z \\
b_x & b_y & b_z
\end{vmatrix}
$$
其中$\mathbf{i} ,\mathbf{j} ,\mathbf{k} $分别对应平行于x、y、z轴的单位向量。通过展开计算可以得出：
$$\mathbf{A} \times \mathbf{B} =(a_yb_z-a_zb_y,a_zb_x-a_xb_z,a_xb_y-a_yb_x)$$

这个向量的方向为根据右手定则向量AB的法向量方向，模长为$||\mathbf{A}|||\mathbf{B}||sin\theta$，其中$\theta$为向量AB夹角大小，即为AB围成的平行四边形的面积大小。

当我们把讨论的对象限定为二位平面上的一组向量时，这个方法便可以指示两个向量之间的方向。
$\overrightarrow{AB} \times \overrightarrow{AC} > 0  => x_{AB}y_{AC}-x_{AC}y_{AB} > 0  => (B_x - A_x)(C_y-A_y)-(C_x-A_x)(B_y-A_y)>0 =>$  AB到AC是逆时针方向
若为0 则说明ABC三点共线 若小于0 则说明顺时针方向。

Andrew算法就是贪心地维护一个类似栈的结构，其中新待加入的点只要和栈顶的两个点组成的三点结构的CROSS(向量外积)<0，则说明AB到AC是逆时针方向的，说明B到C是左转，能维持凸包结构。若取等号，则该凸包可以包括凸包边上的点，而不取等号则不包括边上的点。

在以下情况时，新点将被push进栈：
(1) 栈顶两点和新点的CROSS >(有时候取=)0.
(2) 栈内元素不足2.

当栈顶两点和新点的CROSS不满足条件时，栈顶元素将被pop出去，并重复执行以上操作，直到所有的点都已经完成操作。可以证明，至少有两个点时，左下和右上的两个点至少都会被包含在下凸包中。

构建完下凸包后，我们从右上角的点开始构建上凸包，但此时方向是按从大到小遍历的，并且CROSS的条件改为小于或等于0.

CROSS即为求向量AB和AC的外积的符号。

代码如下：

```cpp
struct Point {
    int x, y;
    bool operator==(const Point &p) const {
        return x == p.x && y == p.y;
    }
};

// 比较函数：先按 x 升序，若相等则按 y 升序
bool cmp(const Point &a, const Point &b) {
    return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
}

// 叉积：判断 A->B->C 的转向
long long cross(const Point &A, const Point &B, const Point &C) {
    return 1LL * (B.x - A.x) * (C.y - A.y) - 1LL * (B.y - A.y) * (C.x - A.x);
}

vector<Point> convexHull(vector<Point> &pts) {
    int n = pts.size();
    if (n <= 1) return pts;

    sort(pts.begin(), pts.end(), cmp);
    vector<Point> hull;

    // 构建下凸包
    for (auto &p : pts) {
        while (hull.size() > 1 && cross(hull[hull.size()-2], hull.back(), p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // 构建上凸包
    int lowerSize = hull.size();
    for (int i = n - 2; i >= 0; i--) {
        while (hull.size() > lowerSize && cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0) {
            hull.pop_back();
        }
        hull.push_back(pts[i]);
    }

    // 去掉最后一个点（起点重复）
    hull.pop_back();

    return hull;
}
```
由于除了第一个点，每个点最多进栈出栈一次，所以遍历过程的时间复杂度为$O(n)$
加上排序所需的时间，总时间复杂度为$O(nlogn)$。

这道题还没有结束，我们求出了凸包以后又怎么求得最大三角形的面积呢？

利用凸包+旋转卡壳即可：

我们依然遍历i、j，可以观察到ij固定时，k只要为三角形面积的极大值点就为最大值点（否则不满足凸包），此外k点也是随着j的增大而增大的，这也是因为凸包的凸多边形性质。因为往回退高必然会变小。

这样，我们只需要遍历ij两层循环，利用双指针即可。这样解答的时间复杂度为$O(n^2)$.

(这是一道真正意义上的困难题)

## Leetcode 976 三角形的最大周长 && 简单贪心

这道题显然可以用 **Leetcode 611**的方法解决，因为能形成三角形的三边必然是满足有效三角形的，而只需要在遍历的过程中维护最大周长即可，但这种方式看起来很浪费时间空间，找一个答案却要花费$O(n^2)$的遍历所有的时间，显然不够聪明。

这题其实是一个非常简单的贪心题，我们只要找出最大周长的话，其实可以考虑以下情况：我们固定三角形的最大边，则最有可能满足剩下两条边a+b>c的只可能是比它小的最大的两条边，并且这个时候若满足，则说明此时的三角形周长最大。也就是说，我们只需要由大到小遍历最大边，如果找到一组满足的三角形则可以直接返回结果了。

这样的算法复杂度为$O(nlogn)$。

## Leetcode 1039 多边形三角剖分的最低得分 && 区间DP

这道题是一个典型的区间DP问题。区间 DP (Interval DP)：
是一类动态规划问题，状态表示的是一个 区间 [l, r] 上的最优解（或某种值）。
通常用在：

括号化问题

多边形划分问题

矩阵链乘法

石子合并、合并果子

最小三角剖分

其核心特征：

问题输入有明显的区间结构（顺序固定，不能随便打乱）。

需要考虑如何把一个区间 [l, r] 分成若干子区间 [l, k] 和 [k, r]。

状态转移就是在不同的分割点 k 之间取最优。

区间DP的状态转移方程为：$$dp[l][r] = min_{l < k < r} ( dp[l][k] + dp[k][r] + cost(l,k,r) )$$

对这道题而言，dp[a][b]表示以a为起始下标，b为最终下标的这个多边形中三角形剖分之和的最小值。

因此，状态转移方程为：
$$ dp[l][r] = min_{l < k < r} ( dp[l][k] + dp[k][r] + values[l] * values[k] * values[r] ) $$

我们对dp函数加一个特判，保证l+1=\=r时返回0，l+2=\=r时返回三边之积即可，同时用一个dp数组保存状态，不至于无穷递归导致指数爆炸。

最终的代码如下：

```cpp
class Solution {
public:
    vector<vector<int>> dp;

    int minSc(const vector<int> & values, int f, int l) {
        if (l - f == 1) return 0; // 两点不能成三角形
        if (l - f == 2) return values[f] * values[f+1] * values[l]; // 三点构成一个三角形

        if (dp[f][l] != -1) return dp[f][l];

        int res = INT_MAX;
        for (int k = f+1; k < l; k++) {
            res = min(res, minSc(values, f, k) + values[f]*values[k]*values[l] + minSc(values, k, l));
        }
        return dp[f][l] = res;
    }

    int minScoreTriangulation(vector<int>& values) {
        int n = values.size();
        dp =  vector<vector<int>>(n, vector<int>(n, -1));
        return minSc(values, 0, n-1);
    }
};

```

## Leetcode 88 合并两个有序数组 && 归并排序

其实，这个问题就是Mergesort中关键操作Merge的变体算法，只不过两个数组不在连续内存空间中。
标准的做法是双指针，分别记录每个数组中当前扫描到的下标并将其填入其中即可。如果从小到大扫描，那么需要额外开一个数组保存第一个数组的原始数值。
如果反过来从最大的数值遍历，则不需要。

但对于Mergesort中的Merge而言，则依然需要额外开一个空间进行保存，这是因为，Mergesort中两个数组的内存空间是连续的，不存在像这道题一样后面填充0的部分，如果不额外开辟空间会导致数据覆盖。