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


## Leetcode 2221 数组的三角和 && 杨辉三角

这道题本质上就是一个模拟题，直接模拟累加就行了。
当然，也可以利用数学观察发现，其实结果中第i个系数的和就是$$\binom{n}{k}=\frac{n!}{k!(n-k)!}$$

只要利用快速算阶乘即可。


## Leetcode 15 三数之和 && 双指针进阶

本题本质上就是，对每个固定的数i，求其他所有的两个数之和为-i的所有组合。我们可以首先对数组进行排序，然后固定一个数，另外一个数从最大值开始搜索。由于固定的那个数是从小到大的，所以另一个数也不可能回退，这样每个数字都只需要寻找一次，把时间复杂度从$O(n^3)$优化到了$O(n^2)$。

## Leetcode 1518 换水问题

小学奥赛题，经典的空瓶换水问题，模拟或者直接用数学求解即可。不卡时间，没难度。

## Leetcode 3100 换水问题II

本质上就是上一题的等差数列版本，没有什么大的区别。

## Leetcode 509 斐波那契数 && 简单DP

如果本题利用递归方法会导致时间复杂度为指数级别。可以直接改写为迭代方法，利用两个数每次保存上次的结果，在下一次运算中用上即可。

## Leetcode 1137 第N个泰波那契数

本题本质上和上一道题是一样的。

## Leetcode 746 使用最小花费爬楼梯

简单的动态规划，需要注意并不是到达最后一个阶梯的值，而是爬上去的值，而爬上去可以从倒数第一个或倒数第二个开始，需要去两者最小值。

## Leetcode 42 接雨水

这是第一道困难题，本质上也是一个动态规划的问题，只要想清楚思路就很简单了。对于每个坐标而言，其能接的水的体积为max(min(左边墙的最大值，右边墙的最大值)-height,0)。

这样，只需要递归地计算每个点的leftMax和rightMax即可。

## Leetcode 407 接雨水II && Dijstra算法 && Priority_queue

本质上，这道题就是接雨水的3D形式，但做法却完全不一样。在2D的接雨水的场景里，雨水只能向左右两个方向流动，因此只需要考虑两边的最大高度即可。但问题是，在3D的场景中，水流可以绕过围墙的方向进行流动，而不能仅仅观察上下左右的围墙最大值简单决定，这就是3D，因此，需要找到一个区域，该区域内被围墙围着，外围围墙的最低高度才是这片区域能装载雨水的最大值。

很显然，所有的最外层节点都不能装水（因为它们的邻居之中有高度为0的节点），因此首先把所有的外部节点加入小根堆之中。
从中选取高度最小的那个 将其周围未被访问邻居的高度更新为max(自身高度，该小节点的水位)

如果该邻居的水位更低 那么说明这个格子被装了水 将装了的水记录到结果中即可。

由于每个节点都会入堆一次出堆一次，最终的时间复杂度为O(mnlogmn)。

其实这个算法很像单源非负边最短路问题中的Dijkstra算法，该算法也是维护一个pq，每次考察堆中离源点最近的边，将其周围未被更新过的邻居更新最近距离并加入堆中。

解决这个问题涉及到C++的一个常见容器priority_queue（优先级队列，即堆）。堆的特性与栈相似，但不同点在于堆顶总是优先级最高的点，因此堆可以保证优先级最高的点先出堆。出堆、入堆的时间复杂度都为$O(logn)$。

priority_queue位于头文件\<queue\>中。一般来说，还可以引入头文件\<functional\>，该文件中有less<> greater<>两类函数可以规定小顶堆和大顶堆。一般来说，less对应的是大顶堆，而greater对应的是小顶堆。（这点与sort函数不同）

priority_queue的常见接口如下：
```cpp
// 默认是大根堆（最大值在顶部）
std::priority_queue<int> pq_max;

// 小根堆（最小值在顶部）
std::priority_queue<int, std::vector<int>, std::greater<int>> pq_min;
//第一个模板参数：存储类型（如 int、pair<int,int> 等）

//第二个模板参数：底层容器类型（通常是 vector 或 deque）

//第三个模板参数：比较函数（默认 less<T> → 大根堆；greater<T> → 小根堆）
pq.push(x) 插入元素x
pq.pop() 删除堆顶元素
pq.top() 访问堆顶元素
pq.empty() 判断堆是否为空
pq.size() 返回当前元素的个数

```

显然，pq的建立需要有一个比较函数来完成对元素的比较，对于自定义数据结构而言需要自定义一套新的比较函数。通常可以采用以下方式：

1.lambda表达式+decltype
```cpp
auto cmp = [](pair<int,int> a, pair<int,int> b) {
    return a.second > b.second; // 按 second 从小到大
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);

pq.push({1,5});
pq.push({2,2});
pq.push({1,3});

while(!pq.empty()) {
    auto p = pq.top(); pq.pop();
    cout << "(" << p.first << "," << p.second << ") ";
}

```
注意这里不能直接像sort函数一样直接返回lambda表达式，因为pq需要的是一个类型。

2.利用struct表示新的比较函数
```cpp
struct cmp {
    bool operator()(const std::pair<int,int>& a, const std::pair<int,int>& b) {
        return a.second > b.second; // second 小的优先
    }
};

std::priority_queue<std::pair<int,int>, std::vector<std::pair<int,int>>, cmp> pq;

```

这道题还有一个优化方法，这个优化方法对于二维数组而言也非常常见：
与其将二维数组存在一个真正的二维数组里，不如用i*n+j表示其坐标然后存在一维数组里，这样可以省去许多麻烦的构造。

完整标答代码如下:
```cpp
class Solution {
public:

    int trapRainWater(vector<vector<int>>& heightMap) {
        int m = heightMap.size();
        int n = heightMap[0].size();
        vector<bool> visited(m*n,false);
        using Node = pair<int,int>; //第一位表示高度,i*n+j表示节点位置
        auto grt = [](const Node &a, const Node &b){
            return a.first > b.first;
        };
        priority_queue<Node, vector<Node>, decltype(grt)> pq; //小顶堆
        //周围一圈先入队
        for(int i=0;i<m;i++)
        {
            if(!visited[i*n])
            {
                visited[i*n]=true;
                pq.push(make_pair(heightMap[i][0],i*n));
            }
            if(!visited[i*n+n-1])
            {
                visited[i*n+n-1]=true;
                pq.push(make_pair(heightMap[i][n-1],i*n+n-1));
            }
        }
        for(int j=0;j<n;j++)
        {
            if(!visited[j])
            {
                visited[j]=true;
                pq.push(make_pair(heightMap[0][j],j));
            }
            if(!visited[n*(m-1)+j])
            {
                visited[n*(m-1)+j]=true;
                pq.push(make_pair(heightMap[m-1][j],n*(m-1)+j));
            }
        }
        int result = 0;
        while(!pq.empty())
        {
            Node tp = pq.top();
            pq.pop();
            int tp_h = tp.first;
            int tp_l = tp.second;
            // 更新周围的四个节点（若有）
            int x = tp_l/n;
            int y = tp_l%n;
            if(x-1>=0 && !visited[(x-1)*n+y])
            {
                if(tp_h>heightMap[x-1][y])
                {
                    result+=tp_h-heightMap[x-1][y];
                    pq.push(make_pair(tp_h,(x-1)*n+y));
                }
                else
                    pq.push(make_pair(heightMap[x-1][y],(x-1)*n+y));
                visited[(x-1)*n+y]=true;
            }
            if(x+1<m && !visited[(x+1)*n+y])
            {
                if(tp_h>heightMap[x+1][y])
                {
                    result+=tp_h-heightMap[x+1][y];
                    pq.push(make_pair(tp_h,(x+1)*n+y));
                }
                else
                    pq.push(make_pair(heightMap[x+1][y],(x+1)*n+y));
                visited[(x+1)*n+y]=true;
            }
            if(y-1>=0 && !visited[x*n+y-1])
            {
                if(tp_h>heightMap[x][y-1])
                {
                    result+=tp_h-heightMap[x][y-1];
                    pq.push(make_pair(tp_h,(x*n+y-1)));
                }
                else
                    pq.push(make_pair(heightMap[x][y-1],x*n+y-1));
                visited[x*n+y-1]=true;
            }
            if(y+1<n && !visited[x*n+y+1])
            {
                if(tp_h>heightMap[x][y+1])
                {
                    result+=tp_h-heightMap[x][y+1];
                    pq.push(make_pair(tp_h,(x*n+y+1)));
                }
                else
                    pq.push(make_pair(heightMap[x][y+1],x*n+y+1));
                visited[x*n+y+1]=true;
            }
        }
        return result;
    }
};
```

此外还有一个优化方式，用数组表示方向，简单来说有两种实现方式：
第一种：单纯的上下左右
```cpp
    int direction[4][2] = {{-1,0},{1,0},{0,1},{0,-1}};

    int dir[5] = {0, 1, 0, -1, 0}; // 5个数，方便成对取
    for (int k = 0; k < 4; k++) {
        int nx = x + dir[k];
        int ny = y + dir[k + 1];
        // 四次循环分别是 (0,1), (1,0), (0,-1), (-1,0)
    }
```
## Leetcode 11 盛最多水的容器

这道题是一个经典的双指针问题，其实可以通过以下的方式理解：
将两个指针固定在0和n-1两边，我们只需要将两端高度更矮的那个向内移动直到两个游标互相触碰，不能再迭代即可。
因为，由于盛水的体积由更低的高度木板和底边距离长度决定，因此向内移动更高的那个永远都不可能再增大容器的容积了（因为矮的那边不会变高，矮边最多等于另一边的高度，而底边的距离是不断缩小的），因此我们不需要遍历每种组合情况就可以计算出盛最多水的容器的组合方式，最多移动n次。

## Leetcode 417 太平洋大西洋水流问题 && BFS/DFS && queue和deque

C++的queue是一种先进先出FIFO的数据结构，位于头文件\<queue\>中，常用于BFS等算法中。
其实与stack的区别就在于queue可以从头尾进行访问，因此支持front()和end()两个不同的接口，而pop()是删掉队首的元素。

deque的全称为double-ended queue（双端队列），位于头文件\<deque\>中与vector不同，是一个可以从两端高效插入和删除的队列。相较于vector只支持push_back，deque同时还支持push_front()操作，因此也能访问front()和back()，同时也支持[]下标进行访问。

这道题的思路很简单，要么从每个点出发进行DFS/BFS观察是否能达到两边边界，但这个需要对每个点做一次BFS/DFS因此复杂度较高。
我们反过来想，由位置关系天然确定的可以流入太平洋/大西洋的点显然位于边界，而且对于其他能够到达这些点的点来说，最终也一定是通过判断能否流入这些点来进行判断的。
因此我们可以反过来想，让水往高处“流动”，反向追溯所有有可能流入边界的数据点即可。这样只需要进行两次BFS。

我们将所有左上边界的点加入BFS的queue中，并做BFS追溯所有可能流入这两个边界的数据点，只有当临近节点的高度值大于等于本节点的数值，说明水能从临近节点流入。
按照这样的方式再对右下边界做一次即可。

BFS是广度优先搜索的简写，简单来说就是维护一个队列，每次把队首未访问（并且满足条件可达）的邻居元素加入队列，遍历完队首后将队首弹出并依次遍历其他的元素即可。因此维护一个queue的数据结构是必要的。

DFS是深度优先搜索的缩写，简单来说就是对某个节点的每个邻居使用DFS，递归访问。这样的访问模式会一遍一遍地深入和回溯，我们需要对每个节点进行标记（是否已经访问），若已经访问则不需要重复访问，最终也可以遍历每个节点。
本质上DFS的算法是基于栈的，而过深的递归深度会导致stack overflow报错，因此我们也可以手动维护一个栈进行迭代的DFS操作，示例代码如下：

```cpp
void dfs_iterative(int x, int y, vector<vector<int>>& grid, vector<vector<bool>>& visited) {
    int m = grid.size(), n = grid[0].size();
    stack<pair<int,int>> st;
    st.push({x, y});
    visited[x][y] = true;

    int dx[4] = {1, -1, 0, 0};
    int dy[4] = {0, 0, 1, -1};

    while (!st.empty()) {
        auto [cx, cy] = st.top();
        st.pop();

        for (int d = 0; d < 4; ++d) {
            int nx = cx + dx[d];
            int ny = cy + dy[d];
            if (nx >= 0 && ny >= 0 && nx < m && ny < n && grid[nx][ny] == 1 && !visited[nx][ny]) {
                visited[nx][ny] = true;
                st.push({nx, ny});
            }
        }
    }
}

```

## Leetcode 778 水位上升的泳池中游泳 && 并查集 && Dijkstra算法变体 && 二分查找

这道题需要我们找到在水位上升的泳池中，能够连通起始点和终点的路径中最小的最大水位是多少，可以看作一个路径的最小化最大值的问题，考虑用类Dijkstra方法解决。每次从堆里权重最小的值的点开始扩展，更新周围可达的所有节点的cost，如果目标节点的高度小于当前节点的cost说明这个节点至少也要cost才能访问到，将其cost更新为本节点的cost，反之则需要等到其高度对应的时间才能访问。

类Dijkstra方法可以解决以下类型的问题：
实际上，Dijkstra 算法是一大类“逐步扩展最优状态”的最短路径问题的模板。
我们可以从“本质特征”出发，来识别哪些问题能归为它的变体。

🧠 一、Dijkstra 的本质思想

Dijkstra 的核心是：

在一张图上，从起点出发，每次扩展当前“代价”最小的点，
并利用该点去更新相邻点的最优代价。
一旦一个点的最优代价确定，它就不会再被更优路径更新。

因此它适用于：

图的边权 非负；

目标是找到某种“代价最小”的路径；

代价满足 单调性（即从起点到某节点的最优代价不会因为路径延长而变小）。

⚙️ 二、可视作 Dijkstra 变体的问题特征

满足以下几个特征的，都可以归入 Dijkstra 思想范畴：

特征	解释	示例
✅ 有图结构	有节点和边，可以定义邻接关系	网格、图、二维地图、状态转移图
✅ 每个边或节点有“代价”	可以是时间、高度、能量、风险等	高度 grid[i][j]、移动耗费
✅ 代价是非负且单调	不会出现“负回报”	移动只会花费时间、增加风险
✅ 想求某种“最小代价”	目标是求最小路径代价或最早可达时间	最短路径、最早抵达、最低风险路线
✅ 状态可以逐步扩展	当前最优状态可以推进到新状态	BFS、堆优化 BFS、最短路问题
🚀 三、常见的 Dijkstra 变体类型

下面我给你列出几类常见的变体类型👇

1️⃣ 最小路径和类

目标：从起点到终点，代价和最小
这是 Dijkstra 的标准用途。

📘 示例：

LeetCode 743. 网络延迟时间

LeetCode 1631. 最小体力消耗路径（边权 = 高度差）

LeetCode 505. 迷宫 II

任意 “加权图最短路径” 问题

2️⃣ 最小化路径上最大值

目标：路径上最大边权最小化
即这条路径上最“难走”的一段尽可能容易。

📘 示例：

LeetCode 778. Swim in Rising Water（游泳的最少时间）

路径代价 = max(经过的格子高度)

LeetCode 1102. Path With Maximum Minimum Value（最大化路径上最小值）

📖 思想：

Dijkstra 把 “代价和” 替换为 “代价 max/min”

每次从堆中取出当前“最好（最小或最大）”的节点继续扩展。

3️⃣ 最早可达类（动态传播）

目标：找出在时间、能量、传播速率上的 最早/最短到达时间。

📘 示例：

信号传播问题（例如网络延迟）

病毒/火焰蔓延问题（传播时间最短）

LeetCode 1368. 使网格图至少代价路径（转弯代价）

📖 思想：

每个节点的“到达时间”不断被更新。

当前堆顶的时间最小，即当前最早能扩展的节点。

4️⃣ 多状态最短路类

目标：节点状态不仅有位置，还有附加维度（例如方向、背包、能量）。

📘 示例：

LeetCode 864. 获取所有钥匙的最短路径

状态 = (位置, 已获得钥匙集合)

LeetCode 847. 访问所有节点的最短路径

状态 = (节点编号, 访问状态mask)

📖 思想：

状态空间仍然是图。

每个状态的代价是最小步数/时间。

依然可用堆扩展“最优状态”。

5️⃣ 风险/概率类

目标：最小化风险，或者最大化成功概率。

📘 示例：

LeetCode 1514. Path with Maximum Probability

每条边有概率 p
想找从 start 到 end 的最大乘积概率路径
取 log 后变为最小化和：-log(p)，可直接 Dijkstra

6️⃣ 特殊约束路径

目标：仍然是最小代价，但路径受某些条件限制（比如转弯次数、能量上限）

📘 示例：

LeetCode 1928. Minimum Cost to Reach Destination in Time

限制“时间 ≤ T”
状态变成 (节点, 时间)
仍可用堆扩展最优状态

堆解法(Dijkstra解法)的代码如下：
```cpp
class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size();
        const int dir[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
        vector<bool> visited(n*n,false);
        using Node = pair<int,int>; //当前代价，位置
        auto gt = [](const Node &a, const Node &b){
            return a.first>b.first;
        };
        priority_queue<Node,vector<Node>,decltype(gt)> pq;
        pq.push({grid[0][0],0});
        visited[0] = true;
        while(!pq.empty())
        {
            auto [cost, pos] = pq.top();
            if(pos==n*n-1)
                return cost;
            pq.pop();
            for(int i=0;i<4;i++)
            {
                int nx=pos/n+dir[i][0],ny=pos%n+dir[i][1];
                if(nx>=0 && nx<n && ny>=0 && ny<n && !visited[nx*n+ny])
                {
                    int newcost = max(cost,grid[nx][ny]);
                    pq.push({newcost,nx*n+ny});
                    visited[nx*n+ny] = true;
                }
            }
        }
        return 0;
    }
};
```

这道题也可以暴力搜索+二分查找的方法，二分查找所需要的水位高度，每次用BFS/DFS暴力搜索即可。

此外，由于每个节点对应的高度值不一样，这道题也可以并查集的方法进行解决。

并查集，简单来说就是支持合并和查找两个快速操作的集合，可以视作一种森林，只需要维护以下数组：
parent/pre[i] 表示下标为i的节点的前驱。

这样，当两个节点相互合并时，就可以把其中一个的递归的前驱(也被称为这个集合的代表元素)绑定为另一个的代表元素即可完成合并。

然而，这样当树的高度太高时可能导致效率变低，因此可以采用路径压缩的方法，在递归查找到树的根节点时，可以将每一个路径上的点的前驱动态更新为根节点，这样就可以优化之后查找的速度。

同时，可以定义一个rank，初始为0，当两个rank相同的集合合并时任取一个rank+1并将另外一个挂载在这个集合上，而其余情况则将rank小的集合挂载在rank大的集合上，这样可以保证更小的集合被挂载在更大的集合上从而提高之后路径压缩的效率。

并查集的一种标准实现如下：

```cpp
    class UnionFind{
        public:
            vector<int> parent;
            vector<int> rank;
            UnionFind(int n) {
                parent = vector<int>(n,0);
                rank = vector<int>(n,0);
                iota(parent.begin(),parent.end(),0);
            }

            int find(int k) {
                if(parent[k]==k)
                    return k;
                parent[k] = find(parent[k]);
                return parent[k];
            }

            void join(int a, int b) {
                a = find(a);
                b = find(b);
                if(a == b)
                    return;
                if(rank[a]>rank[b])
                    parent[b] = a;
                else if(rank[a]<rank[b])
                    parent[a] = b;
                else
                {
                    parent[b] = a;
                    rank[a]++;
                }
                return;
            }

            bool connected(int a, int b)
            {
                return (find(a)==find(b));
            }
    };
```

在这道题目里，可以将能够互相到达的集合视作一个并查集。当水位上涨时，由于每个水位高度唯一对应一个节点，只需要更新该节点周围的边即可，将其周围可达的节点加入这个节点之中。我们在每次更新后检查起始和终结两个节点是否在同一个并查集中即可。

实现代码如下:

```cpp
    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size();
        UnionFind uf = UnionFind(n*n);
        vector<int> idx = vector<int>(n*n,0);
        const int dir[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
            {
                idx[grid[i][j]] = i*n+j;  //这个值对应的位置
            }
        }
        for(int water = 0; water<n*n; water++) //逐个更新，因为每次水位上涨1只会增加对应水位的边
        {
            int x = idx[water]/n, y = idx[water]%n;
            for(int i=0;i<4;i++)
            {
                int nx = x+dir[i][0], ny = y+dir[i][1];
                if(nx>=0 && nx<n && ny>=0 && ny<n && grid[nx][ny]<water)
                    uf.join(x*n+y,nx*n+ny);
            }
            if(uf.connected(0,n*n-1))
                return water;
        }
        return 0;
    }
```

## Leetcode 1488 避免洪水泛滥 && 有序集合set和map

这道题需要我们贪心地找到所有在洪水泛滥之前可以抽空的天数，并将这一天拿来抽水。可以将所有的不下雨（可用于抽水的天数）的日子都利用一个有序的集合Set进行排序，再利用其lower_bound和upper_bound等接口实现贪心查找即可。

Set和Map是C++容器中一类拥有自动排序功能（但不能拥有重复元素），其主要用到的接口和其他访问数据的结构类似：

```cpp
std::set
begin() end()  rbegin() rend()
empty() size()  max_size() clear()
insert(value) emplace(args...) erase(it/key)
find(key) count(key) lower_bound(key) upper_bound(key)
equal_range(key) //返回一个{lower_bound,upperbound} pair
swap(other) 
key_comp() //返回用于排序 key 的比较函数
value_comp() //返回用于排序 value 的比较函数
```

```cpp
std::map
operator[](key) at(key) //若不存在，报异常
insert({key, value})  emplace(key, value)

```

lower_bound返回第一个不小于key的位置，upper_bound返回第一个大于key的位置，如果找不到则返回end()迭代器。

关键点在于，set可以通过lower_bound和upper_bound以及find实现堆某个元素的高效精确/模糊查找（由于set和map底层是红黑树实现的，时间复杂度为O(logn)

multiset和multimap是和set/map的可重复元素版本，也位于头文件\<set\>和\<map\>之中。

## Leetcode 2 两数相加 && 链表

本质上就是一个模拟竖式相加的过程，本质上就是考虑每一位是否有进位，由于加法的进位不会传递两次，只需要每次分别判定即可。

链表可以用一个链表指针next快速指向下一个节点，因此顺序访问速度很快。

## Leetcode 2300 咒语和药水的成功对数

本质上，这道题是需要你进行一次排序，然后对每个对应的spells值找出其在vector中的相对顺位。这个过程可以通过迭代器之间的计算轻松完成。利用lower_bound函数返回迭代器，然后再利用迭代器的运算即可求出对应的位置。

## Leetcode 3494 酿造药水需要的最少总时间 && 流水线DP

这道题需要我们按照流水线处理一些任务，但这些任务之间必须连续完成，因此和传统的流水线不完全一样。

对每个问题而言，它可以开始执行的时间为$$max(这一轮上个任务完成的最早时间，这个流水线空缺出来的最早时间)$$

因此我们需要一个endTime数组来记录上一轮的这个任务什么时候结束

而对于每一次执行的过程而言，可以利用一个cur表示当前已经执行的时间，利用上述逻辑对这个任务所需要的最短时间进行更新。

最终，我们确定了最终更新的时间以后再把时间反向推回去即可。

完整代码如下：

```cpp
class Solution {
public:
    long long minTime(vector<int>& skill, vector<int>& mana) {
        using ll = long long;
        int n = skill.size();
        vector<ll> endTime = vector<ll>(n,0); //上次工作完成的时间
        for(int i=0;i<mana.size();i++)
        {
            ll cur = 0;
            for(int j=0;j<n;j++)
            {
                cur = max(cur,endTime[j]) + 1LL * skill[j]* mana[i];
            }
            endTime[n-1] = cur;
            for(int k=n-2;k>=0;k--)
                endTime[k] = endTime[k+1] - 1LL * skill[k+1] * mana[i];
        }

        return endTime[n-1];
    }
};
```

## Leetcode 239 滑动窗口最大值 && 单调队列

这道题是一道经典的滑动窗口问题。题目需要我们在一串数组中，持续地求出滑动窗口中对应的最大的数值。有一个很直观的想法是维护一个优先级队列pq，其中堆顶存的是堆内的最大值。我们需要把每个节点对应的位置存在对应堆的数组里，我们可以不断地pop()如果当前的堆顶不在我们所求的范围当中，只要在范围中就说明这个数是我们需要求的最大的数。这个算法的时间复杂度为O(nlogn)。

我们还可以用更快的方法来代替这个方法，思考什么样的元素在滑动的过程中才有可能成为最大值。考虑两个都在队列内的元素i,j,其中i小于j且i的值小于j，那么在j加入队列后，i便永远不可能成为滑动窗口内的最大值了，因为j比i晚离开滑动窗口，而j的值又比i大。这样，我们维护一个双端队列deque，保证每次移动窗口后产生新的输入时，都保证当前队尾的元素大于新加入的元素（否则旧的队尾就没有作用了），此时可以很容易地看出队列内的元素是单调递减的。所以，我们只需要返回当前有效的队首元素即可，这个就是我们所求的滑动窗口的最大值。不过，由于滑动窗口的范围有限，我们存储的索引在这个时候可以判断该数是否在对应的滑动窗口之内，从而完成了该题的任务。

C++代码如下:

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        deque<int> que;
        for(int i=0;i<k;i++)
        {
            while(!que.empty() && nums[que.back()]<=nums[i])
                que.pop_back();
            que.push_back(i);
        }
        vector<int> result;
        result.push_back(nums[que.front()]);
        for(int i=k;i<n;i++)
        {
            while(!que.empty() && nums[que.back()]<=nums[i])
                que.pop_back();
            que.push_back(i);
            while(que.front()<=(i-k))
                que.pop_front();
            result.push_back(nums[que.front()]);
        }
        return result;
    }
};
```

## Leetcode 3 无重复字符的最长子串

本题是一个滑动窗口问题，我们记录当前串的起始位置和每一个字符上一次最后出现的位置，当没出现重复字符或者上次出现的位置在当前串起始位置之前，则说明不冲突，更新字符位置和当前串长度；若冲突则直接将当前串的起始位置移到最后出现的位置之后。

由于字符串中字符的种类有限，还可以考虑直接用数组代替哈希表存储这个值，可以让程序的运行效率更快。0-128的数字已经足够cover所需的输入。

## Leetcode 438 找到字符串中所有字母异位词

跟上一道题存储的差不多，由于字母异位词的长度必然和原词一样，可以用滑动窗口遍历访问，每次遍历26个字母即可。进阶一点，使得常数值减小的方法是记录每次改动了哪些字母，记录当前串和目标串共有哪些字母不一样，当diff==0时说明二者相等，这样每次只需要比较对应的两个字母。


## Leetcode 3147 从魔法师身上吸取的最大能量

这道题需要我们求从某个点开始往后间隔为k的序列之和的最大值，由于每k个元素在序列中就会重复在下一个序列中出现一次，因此可以对每个点求后缀和，从后往前寻找最大值即可。

## Leetcode 3186 施咒的最大总伤害

其实这道题本质上就是一个银行打劫问题，被打劫的隔壁的银行不能再次被打劫，这样可以用动态规划来处理这个问题。由于Power的取值很大，不可能对每一个power值进行动态规划，对已有的power值进行统计即可。可以用vector排序之后查找即可。

## Leetcode 3539 魔法序列的数组乘积之和 && 取模运算 && 快速幂 && 数组乘积之和 && 状态DP/剪枝 

这道题相当困难，需要把问题仔细拆解并分析，所以慢慢来，一次把一个问题解决好。

首先，要理解题目的含义。首先给定一个整数m，k，以及一个整数数组nums，nums的长度为n。
一个序列如果满足以下条件被称为魔法序列：
1. seq的序列长度为m
2. $0<=seq[i]<n$ （序列为nums数组的某个下标）
3. $n =  2^{seq[0]}+2^{seq[1]}+...+2^{seq[m-1]}$的二进制形式有k个置位，即pop_count(n)为k

这个序列的数组乘积为$$ prod(seq) = (nums[seq[0]]) * (nums[seq[1]]) * ... * (nums[seq[m-1]])$$
返回所有魔法序列的数组乘积之和，并将答案对 $10^9+7$ 取模。

由于序列的选择可以重复，我们考虑某种特定的排列即可。考虑$c[i]$为0~n-1中i被选择的次数。根据排列组合原理，这样固定的序列的prod为

$$ \frac{m!}{\prod_{i=0}^{n-1}c[i]!} \prod_{i=0}^{n-1}nums[i]^{c[i]}$$




## Leetcode 2273 移除字母异位词后的结果数组

只需要按顺序删除即可，判断字母异位词可以用数组/hashmap或直接判断（由于字符串长度很短）

## Leetcode 3349 检测相邻递增子数组I && Leetcode 3350 检测相邻递增子数组II

这两道题的解法完全一样，只需要求相邻递增子数组的最大长度即可。作为相邻递增子数组有以下两种可能：两个相邻的子数组是接续的，这样相邻子数组的长度为子数组长度的一半向下取整；要么两个相邻的子数组不是接续的，长度为两个相邻递增子数组长度之中更小的那个。

## Leetcode 2598 执行操作后的最大MEX

这道题本质上是让我们找出\[0,value-1\]中所有数字取模之后最小的那个的次数和位置，只需要遍历一遍求得即可。$O(n)$的时间复杂度。

## Leetcode 3003 执行操作后的最大分割数量 && pop_count

难度因子3000+的困难题，但思路可以一步一步拆解。

首先要理解题意，题目给你一个字符串s和整数k

用以下分割操作分割字符串s:

1. 选择 s 的最长 前缀，该前缀最多包含 k 个 不同 字符。
2. 删除 这个前缀，并将分割数量加一。如果有剩余字符，它们在 s 中保持原来的顺序。

而允许你最多改变一处下标对应的字符之后，找到这种情况下的最大分割数量。

考虑暴力解法：即把每一个位置的字符换成另外的，再分别计算分割数量。这种暴力解法的时间复杂度为$O(26*n^2)$，对于这道题目给的数据量是超时的。

因此，考虑能不能用类似DP的方法进行计算：有没有可能修改某个字母不会影响全局的某些分割数量呢？

首先，从前往后统计和从后往前统计算出的分割数量是一样的。因为分割数量只取决于不同字母的相对排列位置。

因此，我们考虑以下情景：

那么对原字符串做出如下划分：以第 i 位为分界，对于左半部分，即第 0 位到第 i−1 位，我们按照从头到尾的方式进行分割，得到的最后一个分割称为第 i 位的左相邻分割，简称为左分割，左分割以前的部分称为前缀分割；而对于右半部分，即第 i+1 位到第 n−1 位，我们按照从尾到头的方式进行分割，得到的最后一个分割称为第 i 位的右相邻分割，简称为右分割，右分割以后的部分称为后缀分割。

于是对于被修改的，位置为 i 的字符，我们只需要考虑其对左分割和右分割的影响，分为以下三种情况：

即使修改了位置为 i 的字符，左分割、右分割内以及第 i 位的不同字符数量仍然不超过 k，左分割、右分割以及第 i 位合并为一个分割，对答案贡献为 1。
左分割的不同字符数量为 k，右分割中不同字符数量也为 k，并且左分割与右分割中不同字符的数量不超过 25，把第 i 位修改为左分割、右分割中不包含的字符后，左分割、右分割以及第 i 位能够重组为三个分割，对答案贡献为 3。
其他情况对答案贡献为 2。
那么我们需要统计在位置 i 处字符的左分割与右分割所包含的信息，包括：前缀分割与后缀分割中包含的分割数量，左分割与右分割的字符掩码以及左分割与右分割中不同字符数量。

C++ AC代码如下：
```cpp
class Solution {
public:

    int pop_count(int a)
    {
        int res = 0;
        while (a > 0)
        {
            a = a & (a - 1);  // 每次消除最低位的 1
            res++;
        }
        return res;
    }

    int maxPartitionsAfterOperations(string s, int k) {
        if(k==26)   return 1;
        int length = s.size();
        vector<vector<int>> left(length,vector<int>(3,0)), right(length,vector<int>(3,0)); // 分别代表前缀/后缀数量，字母掩码和
        int num=0;
        int mask=0;
        int cnt=0;
        for(int i=0; i<length-1; i++)
        {
            int bin = 1 << (s[i]-'a');
            if(!(mask & bin)) //不存在这个字母
            {
                cnt++;
                if(cnt<=k)   //还没形成新的分片
                {
                    mask |= bin;
                }
                else
                {
                    mask = bin;
                    num++;
                    cnt = 1;
                }
            }
            left[i+1][0] = num;
            left[i+1][1] = mask;
            left[i+1][2] = cnt;
        }

        num=0, mask=0, cnt=0;

        for(int i=length-1; i>0; i--)
        {
            int bin = 1 << (s[i]-'a');
            if(!(mask & bin)) //不存在这个字母
            {
                cnt++;
                if(cnt<=k)   //还没形成新的分片
                {
                    mask |= bin;
                }
                else
                {
                    mask = bin;
                    num++;
                    cnt = 1;
                }
            }
            right[i-1][0] = num;
            right[i-1][1] = mask;
            right[i-1][2] = cnt;
        }
        int res = 1;
        for(int i=0;i<length;i++)
        {
            int now_res = left[i][0]+right[i][0]+2;
            int total_mask = left[i][1] | right[i][1];
            int total_count = pop_count(total_mask);
            if(left[i][2] == k && right[i][2] == k && total_count<=25)
                now_res++;
            else if(min(total_count+1,26)<=k)
                now_res--;
            res = max(res,now_res);
        }

        return res;
    }
};
```

pop_count的实现方式用这种方式，可以保证一次消掉最后一个1从而减少移位次数。

1010000 - 1 = 1001111
1010000 & 1001111 = 1000000

## Leetcode 3397 执行操作后不同元素的最大数量

这道题用贪心即可，对于每个数我们都把它尽可能地往最低的位置填充，如果能够填充则把结果+1，注意开始需要对数组进行排序。

## Leetcode 2011 执行操作后变量值

没什么可以讲的，正常模拟。

## Leetcode 1625 执行操作后字典序最小的字符串

## Leetcode 53 最大子数组和

方法一：动态规划