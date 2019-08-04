# week5

## Coding

- [x] K-Means++
  - [c1_kmeanspp.py](c1_kmeanspp.py)/`initial_centroids_kmeanspp(df, k)`
  - 效果见此 [链接](./KMeansResult.md)

## Mathematical

- [ ] ID3 algorithm
  > Therefore your task is to supplyment details of calculation of each node to the tree and provide the evidence why you split a node in such way.  
  
  决策树的内部节点表示一个特征，叶节点表示一个类。构建决策树时，每个内部节点选择一个特征，将输入到这个节点的训练集按该特征值域分裂成若干个子集(分支)。

  选择特征的原则是使各个分裂的训练子集尽可能的 “纯”，即尽可能属于同一类别。

  在 `ID3` 算法中，这个 “纯” 通过信息熵来体现。对每个节点，选择一个使不确定度下降最多的特征，即信息增益最大的特征。

  - 根节点:  
    如同讲义对[14个样本](./data/m1_input.jpg)进行的计算，四个特征的增益分别为：  
    $$\begin{matrix}
      G(D \vert F_{App})=0.246 \\
      G(D \vert F_{Inc})=0.029 \\
      G(D \vert F_{Age})=0.151 \\
      G(D \vert F_{Job})=0.048 \\
    \end{matrix}$$
  
    所以根节点选择增益最大的 `Appearance` 特征进行分裂，将样本分成三类:
    - Ah: [4,5,6,10,14] => 3Y+2N
    - Good: [1,2,8,9,11] => 2Y+3N
    - Great: [3,7,12,13] => 4Y
  - 根节点的`Ah`分支:
    $$\begin{matrix}
      H(D)=-\sum\limits_{k=1}^K{p_klog p_k} = -\frac{2}{5}log(\frac{2}{5})-\frac{3}{5}log(\frac{3}{5})=0.971\\
      H(D \vert F_{App}) = 0.971\\
      H(D \vert F_{Inc}) = 0.951\\
      H(D \vert F_{Age}) = 0.951\\
      H(D \vert F_{Job}) = 0.000\\\\
      G(D \vert F_{App}) = 0.000\\
      G(D \vert F_{Inc}) = 0.020\\
      G(D \vert F_{Age}) = 0.020\\
      G(D \vert F_{Job}) = 0.971\\
    \end{matrix}$$
    - 选择 `Profession` 特征进行分裂，将样本分成两类:
      - Steady: [4,5,10] => 3Y
      - Unstable: [6,14] => 2N
      - `Steady` 分支全为 `Y` 样本，所以其熵为 0，子节点类别为 `Y`
      - `Unstable` 分支全为 `N` 样本，所以子节点类别为 `N`
  - 根节点的`Good`分支:
    $$\begin{matrix}
      H(D)=-\sum\limits_{k=1}^K{p_klog p_k} = -\frac{3}{5}log(\frac{3}{5})-\frac{2}{5}log(\frac{2}{5})=0.971\\
      H(D \vert F_{App}) = 0.971\\
      H(D \vert F_{Inc}) = 0.400\\
      H(D \vert F_{Age}) = 0.000\\
      H(D \vert F_{Job}) = 0.951\\\\
      G(D \vert F_{App}) = 0.000\\
      G(D \vert F_{Inc}) = 0.571\\
      G(D \vert F_{Age}) = 0.971\\
      G(D \vert F_{Job}) = 0.020\\
    \end{matrix}$$
    - 选择 `Age` 特征进行分裂，将样本分成两类:
      - Older: [1,2,8] => 3N
      - Younger: [9,11] => 2Y
      - `Older` 分支全为 `N` 样本，子节点类别为 `N`
      - `Younger` 分支全为 `Y` 样本，子节点类别为 `Y`
  - 根节点的`Great`分支:
    - 全为 `Y` 样本，子节点类别为 `Y`.
  ___

- [ ] C4.5, CART
  - [ ] What is Gain Ratio?
  - [ ] Why we are prone to use Gain Ratio?
  - [ ] How to split a node by using Gain Ratio?
  - [ ] What Gini Index?
  - [ ] How to split a node by using Gini Index?
  - [ ] Why people are likely to use C4.5 or CART rather than ID3?

## Reading

- [ ] AdaBoost
  - [ ] What is AdaBoost algorithm
  - [ ] a. What is a weak classifier?
  - [ ] b. What is a strong classifier?
  - [ ] c. How to combine those weakclassifiers?
  - [ ] d. How to update a weak classifier?
  - [ ] e. How to update the strong classifier?
  - [ ] f. Can you complete the mathematical derivation by hand?
- [ ] Haar Feature
  - [ ] What is Haar Feature
  - [ ] g. What is a Haar feature?
  - [ ] h. Can you find out any upgrade Haar features besides the original one?
  - [ ] i. Can you implement a Haar feature in Python or C++?
  - [ ] j. Can you implement the algorithm in a accelerated way? Like integral image?
  - [ ] k. How to combine Haar feature with AdaBoost?
