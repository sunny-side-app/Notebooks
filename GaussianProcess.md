### ガウス過程と機械学習 ([3]) の概要

#### 1章

　本章は主に線形回帰モデルの復習である。線形回帰モデルの問題点は、例えばある状況で各データ点に対する特徴ベクトルを並べた行列 (計画行列) の転置とそのオリジナルの積の逆行列が不安定になることや、変数の数がデータ点の数よりも多いときに同様の不安定性が見られることである。前者について典型的な状況は、計画行列のある列と別の列に線形の関係があるときである。このときリッジ回帰を使う。元の損失関数 (残差平方和) に正則化項を追加したものを新たな損失関数とする。問題は計画行列の対角成分に正数が加えられることで解決される。

#### 2章

　2章は主にガウス分布の復習である。確率モデルとしての線形回帰モデルの解釈から始まり、リッジ回帰モデルの確率モデル的解釈ー例えばリッジ回帰などの手法で損失関数に追加される正則化項に掛かるチューニングパラメータの確率モデル的意味 (線形のパラメータの分散の対するノイズの分散の比) ーが述べてある。変数たちが多次元ガウス分布に従うとき、その変数たちの一部の次元を固定したときの分布もやはりガウス分布になるが、本章の後半では例えばその公式が導出される。

#### 3章

　3章はガウス過程の導入がなされる。まずガウス過程を定義する。どんな自然数 $N$ についてもd、入力 $\bold{x}_1, \ldots, \bold{x}_N \in \mathcal{X}$ に対応する出力のベクトル $\bold{f} = (f(\bold{x}_1), \ldots, f(\bold{x}_N))$ が平均 $\boldsymbol \mu := (\mu(\bold{x}_1), \ldots, \mu(\bold{x}_N)), K_{i j} = k(\bold{x}_i, \bold{x}_{j})$ を要素とする行列 $\bold{K}$ を共分散行列とするガウス分布 $\mathcal{N} (\boldsymbol \mu, \bold{K})$ に従うとき、**$f$ はガウス過程に従う**といい、これを $f ~\text{～}~ \mathrm{GP} (\mu(\bold x), k(\bold{x}, \bold{x}'))$ と書く。　

　次にガウス過程を用いて回帰問題を解くことを考える。入力 $\bold{x} \in \mathcal{X}$ と出力 $y \in \R$ の $N$ 個のペア
$$
\mathcal{D} = \{(\bold{x}_1, y_1), \ldots, (\bold{x}_N, y_N)\}
$$
が与えられているとする。このとき、 $\bold{x}$ と $y$ の間に次のような関係式があると仮定する。
$$
y_i = f(\bold{x}_i) + \varepsilon_i
$$
誤差 $\varepsilon_i$ については、次のような仮定をおく。
$$
\varepsilon ~\text{～}~ \mathcal{N} (0, \sigma^2)
$$
式 $(1)$ と $(2)$ を仮定すると、 $\bold{y}:=(y_1, \ldots, y_N)$ の同時分布 (観測モデルと呼ぶ, p95) は次のように表せる：
$$
p(\bold{y} | \bold{f}) = \mathcal{N} (\bold{f}, \sigma^2 \bold{I})
$$
ここで $\bold{f} := (f(\bold{x}_1), \ldots, f(\bold{x}_N))$ である。 $\bold{f}$ にはガウス過程に従う仮定をおく (本文で明示されていないが必要)：
$$
\bold{f}|\bold{X} ~\text{～}~ \mathcal{N} (\mu, \bold{K})
$$

$\bold{X} := (\bold{x}_1, \ldots, \bold{x}_N)$ である。 $\bold{X}$ が与えられた後の $\bold{y}$ の分布は次のようになる：
$$
\begin{align}
p(\bold{y} | \bold{X}) & = \int p(\bold{y}, \bold{f} | \bold{X}) d\bold{f} \\
    & = \int p(\bold{y} | \bold{f}) p(\bold{f} | \bold{X}) d \bold{f} \\
    & = \int \mathcal{N} (\bold{y} | \bold{f}, \sigma^2 \bold{I}) \mathcal{N} (\bold{f} | \boldsymbol \mu, \bold{K}) d \bold{f} \\
    & = \mathcal{N} (\boldsymbol \mu, \bold{K} + \sigma^2 \bold{I}) \\
\end{align}
$$
$(6)$ から $(7)$ への変形は、 次のように分解できることからわかる：
$$
p(\bold{y}, \bold{f} | \bold{X}) = \frac{p(\bold{y}|\bold{f}, \bold{X}) p(\bold{f}, \bold{X})}{p(\bold{X})}
$$
$(7)$ から $(8)$ への変形は、観測モデルの仮定 $(4)$ と $\bold{f} | \bold{X}$ に対するガウス過程の仮定 $(5)$ からわかる。 $(8)$ から$(9)$ への変形は、ガウス分布の連鎖則の公式による。これは[1] からの引用である (次に説明する)。

$p(x) ~\text{～}~ \mathcal{N} (x|\mu, \Lambda^{-1})$, $p(y|x) ~\text{～}~ \mathcal{N} (y|Ax + b, L^{-1})$ としたとき、次が成り立つ：  
$$
\begin{align}
p(y) & = \int p(x, y) dx \\
& = \int p(x) p(y|x) dx \\
& = \mathcal{N} (y|A\mu + b, L^{-1} + A\Lambda^{-1}A^{\top})
\end{align}
$$
この公式において、$A=\bold{I}, b=\bold{0}$ とおけば元の状況になる。

　ガウス過程回帰モデルのデータ点 $\bold{x}^*$ における予測を考える。線形モデルではパラメータ $w$ と特徴ベクトル $\phi(\bold{x}^*)$ を用いて次のように予測することができた：
$$
y^* = w^{\top} \phi (\bold{x}^*)
$$
しかし、ガウス過程回帰モデルの場合は $\bold{y}$ に $y^*$ を含めたものを新しく $\bold{y}'$ とし、同様の観測モデルの仮定$(4)$ と $\bold{f}$ に対するガウス過程の仮定 $(5)$ をおく。結果として、これら全体がまたガウス分布に従うことになる：
$$
\bold{y}' | \bold{X} ~\text{～}~ \mathcal{N} (\boldsymbol \mu', \bold{K}')
$$
ここで、 $\boldsymbol \mu'$ は元の平均ベクトルに $\mu(\bold{x}^*)$ を追加した期待値ベクトルであり、$\bold{K}'$ は次のような$(N+1) \times (N+1)$ のカーネル行列である：
$$
\bold{K}' = \begin{pmatrix}
\bold{K} & \bold{k}_{*} \\
\bold{k}_{*}^{\top} & k_{**} \\
\end{pmatrix}
$$
ここで、$\bold{k}_{*}$ と $k_{**}$ の定義は次である：
$$
\begin{cases}
	\bold{k}_{*} = (k(\bold{x}^{*}, \bold{x}_1), k(\bold{x}^{*}, \bold{x}_2), \ldots, k(\bold{x}^{*}, \bold{x}_N))^{\top} \\
	k_{**} = k(\bold{x}^{*}, \bold{x}^{*})
\end{cases}
$$
　次に、多次元ガウス分布の条件付き分布 (次に示す) を用いて $(15)$ 式からガウス過程回帰の予測分布を導く。

$\left(\begin{array} \bold x_1 \\ \bold x_2 \end{array}\right) \text{～} \mathcal{N}\left( \left(\begin{array} \bold 0 \\ \bold 0 \end{array}\right), \left(\begin{array} \boldsymbol \Sigma_{11} & \boldsymbol \Sigma_{12} \\ \boldsymbol \Sigma_{21}  & \boldsymbol \Sigma_{22} \end{array}\right) \right) $ とし $\left(\begin{array} \boldsymbol \Lambda_{11} & \boldsymbol \Lambda_{12} \\ \boldsymbol \Lambda_{21}  & \boldsymbol \Lambda_{22} \end{array}\right) = \left(\begin{array} \boldsymbol \Sigma_{11} & \boldsymbol \Sigma_{12} \\ \boldsymbol \Sigma_{21}  & \boldsymbol \Sigma_{22} \end{array}\right)^{-1}$ とおいたとき
$$
\begin{align}
    p(\bold{x}_2|\bold{x}_1) & \propto p(\bold{x}_1, \bold{x}_2) \\
    & \propto \exp\left( -\frac{1}{2}\left\{ \left(\begin{array} \bold{x}_1 - \boldsymbol \mu_1 \\ \bold{x}_2 - \boldsymbol \mu_2 \end{array}\right)^{\top} \left(\begin{array} \boldsymbol \Lambda_{11} & \boldsymbol \Lambda_{12} \\ \boldsymbol \Lambda_{21} & \boldsymbol \Lambda_{22} \end{array}\right) \left(\begin{array} \bold{x}_1 - \boldsymbol \mu_1 \\ \bold{x}_2 - \boldsymbol \mu_2 \end{array}\right) \right\} \right) \\
    & \propto \exp \left(-\frac{1}{2}\{(\bold{x}_1 - \boldsymbol \mu_1)^\top \boldsymbol \Lambda_{11} (\bold{x}_1 - \boldsymbol \mu_1) + (\bold{x}_1 - \boldsymbol \mu_1)^\top \boldsymbol \Lambda_{12} (\bold{x}_2 - \boldsymbol \mu_2) \\
     + (\bold{x}_2 - \boldsymbol \mu_2)^\top \boldsymbol \Lambda_{21} (\bold{x}_1 - \boldsymbol \mu_1) + (\bold{x}_2 - \boldsymbol \mu_2)^\top \boldsymbol \Lambda_{22} (\bold{x}_2 - \boldsymbol \mu_2)\} \right) \\
    & \propto \exp \left(-\frac{1}{2}\{(\bold x_2 - \boldsymbol \mu_2)^\top \boldsymbol \Lambda_{22} (\bold{x}_2 - \boldsymbol \mu_2) + 2 (\bold{x}_1 - \boldsymbol \mu_1)^\top \boldsymbol \Lambda_{21} (\bold{x}_2 - \boldsymbol \mu_2)\} \right) \\
    & \propto \exp \left(-\frac{1}{2} \{\bold x_2^\top \boldsymbol\Lambda_{22} \bold x_2 -\bold x_2^\top \boldsymbol \Lambda_{22} \boldsymbol \mu_2 - \boldsymbol \mu_2^\top \boldsymbol \Lambda_{22} \bold x_2 +2(\bold x_1 - \boldsymbol \mu_1)^\top \boldsymbol \Lambda_{21} \bold x_2 \}\right) \\
    & = \exp \left( -\frac{1}{2} \{ \bold x_2^\top \Lambda_{22} \bold x_2 -2 \bold x_2^\top(\boldsymbol \Lambda_{22} \boldsymbol \mu_2 - \boldsymbol \Lambda_{21}(\bold x_1 - \boldsymbol \mu_1))\} \right) \\
    & \propto \exp \left( -\frac{1}{2} \{(\bold x_2 - \boldsymbol \Lambda_{22}^{-1}(\Lambda_{22} \boldsymbol \mu_2 - \boldsymbol \Lambda_{21}(\bold x_1 - \boldsymbol \mu_1)) )^\top \boldsymbol \Lambda_{22} (\bold x_2 - \boldsymbol \Lambda_{22}^{-1}(\Lambda_{22} \boldsymbol \mu_2 - \boldsymbol \Lambda_{21}(\bold x_1 - \boldsymbol \mu_1)) )  \} \right)
\end{align}
$$

と平方完成できるため、 $\bold x_2$ の分布は、 $\boldsymbol \Lambda$ で表すと次のようになる。
$$
p(\bold x_2 | \bold x_1) ~\text{～}~ \mathcal{N} (\boldsymbol \mu_2 - \boldsymbol\Lambda_{22}^{-1}\boldsymbol\Lambda_{21}(\bold x_1 - \boldsymbol \mu_1), \bold\Lambda_{22}^{-1})
$$
あとはこれを $\boldsymbol \Sigma$ を使って表すだけである。ブロック行列 $\left(\begin{array} \boldsymbol A & B \\ C  & D \end{array}\right)$ の公式を用いる (次に説明する)。

$A^{-1}, D^{-1}$ が存在するならば、一般に次の関係が成り立つ。

 $M:=(A - BD^{-1}C)^{-1}$ のとき、
$$
\left(\begin{array} \boldsymbol A & B \\ C  & D \end{array}\right)^{-1} = \left(\begin{array} M & -MBD^{-1} \\ -D^{-1}CM & D^{-1} +D^{-1} CMBD^{-1} \end{array} \right)
$$
 $M:=(D - CA^{-1}B)^{-1}$ のとき、
$$
\left(\begin{array} \boldsymbol A & B \\ C  & D \end{array}\right)^{-1} = \left(\begin{array} A^{-1} + A^{-1}BMCA^{-1} & -ABM^{-1} \\ -MCA & M \end{array} \right)
$$
したがって、 $M:=(\Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{21})^{-1}$ とおくと次が成り立つ。
$$
\begin{align}
\left(\begin{array} \boldsymbol \Lambda_{11} & \boldsymbol \Lambda_{12} \\ \boldsymbol \Lambda_{21}  & \boldsymbol \Lambda_{22} \end{array}\right)^{-1} & = \left(\begin{array} \boldsymbol \Sigma_{11} & \boldsymbol \Sigma_{12} \\ \boldsymbol \Sigma_{21} & \boldsymbol \Sigma_{22} \end{array} \right)^{-1} \\
& = \left( \begin{array} \cdots & \cdots \\ -M \boldsymbol \Sigma_{21} \boldsymbol \Sigma_{11}^{-1} & M \end{array} \right)
\end{align}
$$
よって、
$$
\begin{align}
	\boldsymbol \Lambda_{22} & = (\boldsymbol \Sigma_{22} - \boldsymbol \Sigma_{21} \Sigma_{11}^{-1} \boldsymbol \Sigma_{21})^{-1} \\
	\boldsymbol \Lambda_{22}^{-1} \boldsymbol \Lambda_{21} & = - \boldsymbol \Sigma_{21} \boldsymbol \Sigma_{11}^{-1}
\end{align}
$$
となるため、
$$
\begin{align}
p(\bold x_2 | \bold x_1) & ~\text{～}~ \mathcal{N} (\boldsymbol \mu_2 - \boldsymbol\Lambda_{22}^{-1}\boldsymbol\Lambda_{21}(\bold x_1 - \boldsymbol \mu_1), \bold \Lambda_{22}^{-1}) \\
& = \mathcal{N} (\boldsymbol \mu_2 + \boldsymbol \Sigma_{21} \boldsymbol \Sigma_{11}^{-1} (\bold x_1 - \boldsymbol \mu_1), \boldsymbol \Sigma_{22} - \boldsymbol \Sigma_{21} \boldsymbol \Sigma_{11}^{-1}  \boldsymbol \Sigma_{21})
\end{align}
$$

 $(15)$ に多次元ガウス分布の条件付き分布の公式を適用することで、ガウス過程回帰モデルの予測分布が導出される：
$$
p(y^*|\bold x^*, \mathcal{D}) = \mathcal{N}(\mu(\bold x^*) + \bold k_*^\top\bold K^{-1}\bold y, k_{**} - \bold k_*^\top \bold K^{-1} \bold k_*)
$$
上では1点 $\bold x^*$ における出力 $y^*$ の予測分布を考えたが、予測したい点が $\bold X^* = (\bold x^*_1, \ldots, \bold x^*_M)$ と $M$ 個ある場合も議論は同様であり、次のように予測分布が導かれる： 
$$
p(\bold y^*|\bold X^*, \mathcal{D}) = \mathcal{N} (\mu'' + \bold k_*^\top \bold K^{-1} \bold y, \bold k_{**} - \bold k_*^\top \bold K^{-1} \bold k_*)
$$
ただし、$\bold k_* (n,m) = k(\bold x_n, \bold x_m^*), \bold k_{**} (m, m) = k(\bold x_m^{*}, \bold x_m^{*}) (n=1, \ldots, N, m=1, \ldots, M)$ であり、$\mu''$ は $\mu(\bold{x}_1^*), \ldots, \mu(\bold x_M^*)$ をまとめた期待値ベクトルである。

​	ガウス過程には予測分布 $(34)$ の期待値の計算で現れる $\bold K^{-1}$ の計算が $O(N^3)$ の計算量であり、かつてボトルネックであった (が、5章で解説されている補助変数法や変分法によって現在では緩和された)。

　ガウス過程回帰のパラメータ推定について説明する。ガウス過程回帰では、 $(35)$ から分かるように、カーネル関数とそのハイパーパラメータが与えられれば、学習データから予測分布が計算できる。したがって、(ハイパーパラメータのチューニングを除けば) いわゆる「学習」フェーズ (最適化のフェーズ) は存在しない。ガウス過程回帰のハイパーパラメータチューニングは、 $(9)$ の尤度を最大化するように行う。 ハイパーパラメータを $\boldsymbol \theta$ と表すと、尤度は次のように表現できる：
$$
p(\bold y | \bold X, \boldsymbol \theta) = \mathcal{N} (\bold y|\boldsymbol \mu, \bold K_{\boldsymbol \theta})
$$
この対数をとり、MCMC法や勾配法で最適化する。

　ガウス過程回帰の一般化を考える。ここまでのガウス過程回帰では観測モデル $(4)$ にガウス分布を仮定してきた。つまり、観測値 $y$ として $f(\bold x)$ にガウス分布に従う誤差が加わった値が観測されるという場合を考えてきたがこのガウス分布の仮定は現象によってさまざまな分布になりうるという話がされている。この場合、($(6) \text{～} (9)$ では求まっていたが) 解析的に解が求まらず、MCMC法や変分ベイズ法など近似推論を行う必要がある。

#### 4章

　4章では確率的生成モデル (確率モデル) を導入している。例えばガウス過程回帰モデルも、ガウス過程の仮定と観測モデルに対する正規分布の仮定を組み合わせた確率モデルとして見ることができる。その後最尤推定とベイズ推定が導入される。最後に、(実際上) 確率分布を計算する、とはどういうことかが説明される。確率分布を解析的な形で書くのが難しいがその分布からのサンプルは計算機で容易に得られるというとき、(ヒストグラムなどで) サンプルを可視化するのが分布の概形を把握する上で有効である。また、この場合、(記載は無いが大数の法則によって) 期待値や分散などの統計量も計算できる。その分布の値は計算機で容易に計算できるが、サンプルを得ることも難しいというときにも、重みつき標本を使って近似する方法がある (Importance sampling のことと思われる)。

#### 5章

　5章では補助変数法と変分ベイズ法が説明される。ガウス過程回帰のボトルネックは (3章で述べたように) 予測分布 $(34)$ の期待値の計算で現れる $\bold K^{-1}$ の計算量であった。部分データ法は $N$ が大きすぎるときに、データ点全体をうまく代表するようなこのうちのいくつかを選んで残りを捨てるという方法である。補助変数法はその延長にあるようである (捨てるデータ点をうまく使うようだが全体的によくわからなかった)。変分ベイズ法は、ベイズ推定において事後分布を近似する手法である。ベイズ推定法では一般的に事後分布を計算し、ハイパーパラメータを周辺尤度 $p(Y|\theta) = \int p(Y|\bold w, \theta) p(\bold w|\theta)d\bold w$ を最大化するように求める ( $Y$ は観測、$\bold w$ はパラメータ、$\theta$ はハイパーパラメータ)。素朴には最適なハイパーパラメータをグリッドサーチで探索するなどの方法があるが、グリッドの候補が多い場合にはこれは計算的に難しいため、事後分布を別のパラメトリックな分布 (=変分事後分布) で近似するのが変分ベイズ法である。つまり、事後分布のモデルを新たに考え、それは厳密には事後分布と一致しない可能性はあるが、計算的に扱いやすいものとなる。変分事後分布のパラメータを推定するためには、エビデンス (周辺尤度) の下界を考え (変分事後分布のパラメータと元々のハイパーパラメータに依存する)、これを最大化するように推定する。

#### 6章

ベイズ最適化が紹介されている。ベイズ最適化では、未知だが入力を与えたときの出力のみは分かるブラックボックス関数 $f(\bold x)$ があり、それを最大化する $\bold x^*$ がほしいという目的がある。入力を与え出力を得るという「実験」を行いデータ点 $(\bold x_1, y_1), \ldots, (\bold x_N, y_N)$ が得られたとして、このブラックボックス関数 $f(\bold x)$ にガウス過程を仮定すると、期待値 $\mu(\bold x)$、標準偏差 $\sigma(\bold x)$ のベイズ予測分布が計算できる。 $f(\bold x)$ を最大化する $\bold x$ を考えると、期待値を最大化する領域の $\bold x$ が良さそうであるが、実験があまり行われていない特徴の領域 (= $\sigma (\bold x)$ が大きい領域) も良さそうである。そこで、これら $\mu$ と $\sigma$ を組み合わせた獲得関数というものを考え、これを最大化するように $\bold x_{N+1}$ を決める。この獲得関数にもいくつか種類がある。

### 補足

ガウス過程回帰モデルのあるデータ点における予測は、そのデータ点が学習データから離れていくにつれてある点に収束していくという現象が見られる。これを示す (非常に簡単)。

ガウス過程回帰モデルのある点 $\bold x^*$ における予測分布は $(34)$ 式で与えられていた：
$$
p(y^*|\bold x^*, \mathcal{D}) = \mathcal{N}(\mu(\bold x^*) + \bold k_*^\top\bold K^{-1}\bold y, k_{**} - \bold k_*^\top \bold K^{-1} \bold k_*) \notag
$$
ここで、$\bold x^* \to \infty$ の極限を考えると、$\bold k^* \to \bold 0$ より $\mu(\bold x^*) + \bold k_*^\top\bold K^{-1}\bold y \to \mu (\bold x^*)$、$k_{**} \to 1$ より $k_{**} - \bold k_*^\top \bold K^{-1} \bold k_* \to 1$ となる。 

### 所感、重要だと思ったポイントなど

- 全体的に数式とその (直観的) 意味が対応する形で述べられており、数式からすぐに分からない自分のようなタイプには非常に良い本だと思った。例えば1章のリッジ回帰の正則化パラメータの意味や2章の多変量ガウス分布の公式の意味など。また、さまざまな概念との関連が述べられており、少しだけいくつかの分野を俯瞰できるようになった気がする (気がする)。

- カーネル法との関連について述べてある箇所があり、違いが分かって良かった。本書3章によれば、カーネル法では予測分布の期待値がデータ点 (の非線形変換) の線形結合で表される (= リプレゼンタ定理) という性質を利用しており、この点はガウス過程回帰も共通しているが、その重み自体をカーネル法では最適化して求めているのに対し、(ガウス過程回帰のパラメータ推定の箇所で述べたように) ガウス過程回帰では最適化は行わない (行列計算するのみ) ようである。

- カーネル法との関連で言えば、3章でカーネル関数が導入され、特徴ベクトルの内積がカーネル関数の内積だけで計算でき (= カーネルトリック) 特徴ベクトル (= 福水 (2010) の言葉で言えば特徴写像) は陽に導出する必要はないということが述べられたが、そのような特徴写像は存在するのかどうか気になったため、少しだけ調べたところ、福水 (2010) にその記載があった：

  再生核ヒルベルト空間と正定値カーネルが1対1に対応することによって、正定値カーネルを与えることで対応する再生核ヒルベルト空間が定まり、それを終域とする特徴写像 $\Phi$ が定義でき、その結果再生性から $\langle \Phi(x), \Phi(y) \rangle = k(x,y)$ が成り立つ (つまり正定値カーネルから特徴写像を定められる) (福水, 2010, p. 18-19)。ちなみに、これを示すために必要な定理としては、命題2.6, 命題2.7, 定理2.11 (福水, 2010)  が挙げられている。

### 疑問

- これまで全く疑問に思わなかったが、各データ点の同時分布を考えるということが (よくやるが) いまいち理解できていない気がする。例えばこの分散をデータから計算するとすれば、どうやって計算するのか？ (多変数の同時分布の共分散とかなら計算できるし、各データ点の同時分布の尤度とかもよく計算するが)
- 4章でベイズ推定の目的は事後分布を求めることですと記載があったが、ベイズ推定のモチベーション、特に変分ベイズ法のモチベーションが整理できていないのを感じた。ベイズ推定には (事後分布計算後に) ベイズ予測分布がほしいというモチベーションと周辺尤度を最大化したいというモチベーションがあると思っており、周辺尤度最大化はハイパーパラメータの最適化 (例えば事前分布のパラメータなど) のために行うのだという認識があった。しかし、まず4章でハイパーパラメータを (周辺尤度ではなく) 尤度を最大化するように推定しており、よくわからなかった。5章の変分ベイズ法では、事後分布をKL div. の意味で近似するパラメトリックなモデルを考えますという導入はついていけたが、その後なぜ周辺尤度の下界を考えるのか、そしてなぜそれを最大化するハイパーパラメータがほしかったハイパーパラメータになるのか (元々ハイパーパラメータは周辺尤度 (または本書4章によれば尤度) を**最大化**するように推定するのではなかったか？) というところが分からず、それまでの流れとのつながりが分からなかった。
- ガウス過程回帰モデルの予測分布を求める際に、(線形回帰ではなく) ガウス過程では $w ~\text{～}~ \mathcal{N} (\bold{0}, \lambda^2 \bold{I}) (\lambda > 0)$ を仮定しており、その線形変換である $\bold{y} = \Phi w$ もガウス分布 $\mathcal{N} (\bold{0}, \lambda^2 \Phi \Phi^{\top})$ に従うため、パラメータを求める必要がない ($w$ は期待値の計算で積分消去される) ために困ってしまう (が問題ない) というコメント (意訳) があったが、そもそもガウス過程回帰モデルではパラメータは考えていない (つまり考えているモデルが違う) のではと思った。自分が抱えている大きな疑問の1つは、線形回帰モデルからのガウス過程への導入の流れから、観測ノイズの導入部分 (3.3.3節) で唐突にガウス過程に従う $f$ が出現して (つまりノイズの項が加わる以外にもモデルが変化しており) 、それまでの線形回帰モデルの拡張としてのガウス過程とのつながりが分からなくなったところである。線形回帰ではパラメトリックモデルを考えているがガウス過程ではノンパラモデルを考えている (ゆえに予測 $y^*$ がパラメータの線形で表現できない) という違いではないのかと思った。

#### 参考文献

[1] Bishop, C. M. (2006). *Pattern recognition and machine learning*. springer.

[2] 福水健次. カーネル法入門. 朝倉書店. 2010.  

[3] 持橋大地, 大羽成征. ガウス過程と機械学習. 講談社. 2019.  