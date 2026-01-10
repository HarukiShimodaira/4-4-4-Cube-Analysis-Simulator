# 回転行列の仕様（数学的定義）

このプロジェクトの回転は、`src/core/matrix3x3.py`, `src/core/cubelet.py`, `src/core/cube4x4.py` の実装に従います。

# 免責
このドキュメントはGithub Copilotを使用して生成しており、正確性は担保されません。

## 1. 座標系（右手系）

空間は $\mathbb{R}^3$。座標軸は次で固定です。

- $+x$: 右（R側）
- $+y$: 上（U側）
- $+z$: 前（F側）

## 2. 回転行列 $R$ の定義（$SO(3)$）

回転行列 $R\in\mathbb{R}^{3\times 3}$ は次を満たす行列です。

- 直交性: $R^\top R = I$
- 右手系の回転（特殊直交）: $\det R = 1$

実装上も `Matrix3x3.is_orthogonal()` と `Matrix3x3.is_proper_rotation()` で同様に検証します。

## 3. ベクトルの表現（列ベクトル）と作用

位置・方向ベクトル $v\in\mathbb{R}^3$ は **列ベクトル**として扱い、回転の作用は

$$
v' = R\,v
$$

です（`Matrix3x3.apply_to_vector()` は `m @ v`）。

## 4. 合成の順序

回転 $A$ の後に $B$ を適用する合成は

$$
v'' = B(Av) = (BA)\,v
$$

つまり、**後から適用する回転ほど左側に掛かる**という列ベクトルの標準的な規約です。

## 5. キューブレット姿勢の更新規約（ワールド座標系で左作用）

層回転の回転行列を $Q$、キューブレットの現在姿勢を $R$ とすると、更新は

$$
R_{\text{new}} = Q\,R_{\text{old}}
$$

です（`Cubelet.apply_rotation()` が `rotation_matrix @ self.rotation`）。

位置（中心座標）も同様に

$$
p_{\text{new}} = Q\,p_{\text{old}}
$$

で更新します。

## 6. 軸回転行列の具体形（右手系）

`Matrix3x3.rotation_x/y/z(theta)` の `theta` は **度数法**で、内部でラジアンへ変換して標準形を使います。

$$
R_x(\theta)=\begin{pmatrix}
1&0&0\\
0&\cos\theta&-\sin\theta\\
0&\sin\theta&\cos\theta
\end{pmatrix},\quad
R_y(\theta)=\begin{pmatrix}
\cos\theta&0&\sin\theta\\
0&1&0\\
-\sin\theta&0&\cos\theta
\end{pmatrix},\quad
R_z(\theta)=\begin{pmatrix}
\cos\theta&-\sin\theta&0\\
\sin\theta&\cos\theta&0\\
0&0&1
\end{pmatrix}.
$$

## 7. ルービック記法（R/U/F 等）の「時計回り」と角度符号

`Cube4x4` では、各手の定義が

- 「その面を外側から見たときの時計回り/反時計回り」

になるように、\(\pm 90^\circ\) の符号を面ごとに調整しています。

例（`src/core/cube4x4.py`）:

- `R`: x=+1.5 層を $-90^\circ$（外側から見て時計回り）
- `Rp`: x=+1.5 層を $+90^\circ$

これは「右手系での正回転（+軸方向から原点を見る）」と「面を外側から見た時計回り」の基準が、面によって一致/反転するためです。

## 8. 「初期→現在」の相対回転 $R_{\mathrm{rel}}$

初期姿勢 $R_0$、現在姿勢 $R$ に対し、初期からの差分回転は

$$
R_{\mathrm{rel}} = R\,R_0^\top
$$

です。

これは

$$
R_{\mathrm{rel}}\,R_0 = R
$$

を満たし、「初期姿勢に差分回転をかけると現在姿勢になる」ことを意味します。

`src/analysis/random_data_collector.py` ではこの $R_{\mathrm{rel}}$ から回転角・回転軸を推定します。

補足（実装上の軸の推定）:
- `RandomDataCollector` では $R_{\mathrm{rel}}$ から
  - $\theta$ はトレース公式
  - 軸は $[R_{32}-R_{23},\; R_{13}-R_{31},\; R_{21}-R_{12}]$ を正規化
  という形で推定します。
- 上記ベクトルは厳密には $2\sin\theta\,\hat{n}$ に比例するため、正規化で軸方向 $\hat{n}$ を得る方針です。

## 9. 角度の取り出し（トレース公式）

回転行列 $R\in SO(3)$ の回転角 $\alpha\in[0,\pi]$ は

$$
\alpha = \arccos\left(\frac{\operatorname{tr}(R)-1}{2}\right)
$$

で与えられます。

実装では数値誤差対策として $(\operatorname{tr}(R)-1)/2$ を $[-1,1]$ に `clip` してから `arccos` し、最後に度数法へ変換しています。

---

### 実装参照

- `src/core/matrix3x3.py`
  - `rotation_x/y/z()`（度数法、右手系）
  - `apply_to_vector()`（$v' = Rv$）
  - `__matmul__()`（行列積）
- `src/core/cubelet.py`
  - `apply_rotation()`（$p' = Qp$, $R' = QR$）
- `src/core/cube4x4.py`
  - `R/Rp/...`（各手の符号と回転層）
- `src/analysis/random_data_collector.py`
  - `R_relative = R_current @ R_initial^T`
