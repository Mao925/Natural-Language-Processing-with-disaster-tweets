# Kaggle NLP Binary Classification Baseline

## プロジェクト概要

本プロジェクトでは、Kaggle の NLP 二値分類タスクに対して、まずはシンプルで再現性の高いベースラインモデルを構築した。

目的は、`text` から `target` を予測することである。

今回のベースラインでは、Transformer や深層学習モデルは使わず、以下の構成を採用した。

- 前処理：TF-IDF 向けの軽いテキストクリーニング
- 特徴量化：TF-IDF
- モデル：Logistic Regression
- 提出ファイル：`submission_tfidf_logreg.csv`

まずは高性能なモデルをいきなり作るのではなく、データの構造を理解し、今後の改善基準となるスコアを作ることを重視した。

---

## 使用データ

Google Drive 上に配置した以下の CSV ファイルを使用した。

- `/content/drive/MyDrive/Kaggle/NLP/train.csv`
- `/content/drive/MyDrive/Kaggle/NLP/test.csv`

### train.csv のカラム

| カラム名 | 内容 |
|---|---|
| id | 各データのID |
| keyword | 投稿に紐づくキーワード |
| location | 投稿者の位置情報 |
| text | 投稿本文 |
| target | 目的変数。二値分類ラベル |

### test.csv のカラム

| カラム名 | 内容 |
|---|---|
| id | 各データのID |
| keyword | 投稿に紐づくキーワード |
| location | 投稿者の位置情報 |
| text | 投稿本文 |

---

## タスク設定

このタスクでは、`text` の内容から `target` を予測する。

分類タスクとしては二値分類であり、`target` は以下のような値を取る。

| target | 意味 |
|---|---|
| 0 | クラス0 |
| 1 | クラス1 |

今回の目的は、まず `input_text` から `target` を予測するベースラインを作成することである。

---

## 前処理方針

今回のベースラインでは、Transformer 向けの前処理ではなく、TF-IDF + LogisticRegression 向けの前処理を行った。

TF-IDF では、過剰にテキストを加工しすぎると有用な語彙情報まで失われる可能性がある。そのため、stopwords 除去、stemming、lemmatization などの強い正規化は行わず、軽いクリーニングに留めた。

### 実施した前処理

- `train.csv` と `test.csv` を読み込み
- `keyword` の欠損値を空文字 `""` で補完
- `text` の欠損値も念のため空文字 `""` で補完
- `location` は欠損が多く、ノイズも多いと判断し、最初のベースラインでは使用しない
- `text` に対して軽いクリーニングを実施
  - HTMLエスケープを戻す
  - URLエンコードを戻す
  - 小文字化
  - URLを `<url>` に置換
  - メンションを `<user>` に置換
  - ハッシュタグは `#` のみ削除し、単語自体は残す
  - 連続する空白を1つにまとめる
  - 前後の空白を削除
- `keyword` にも同じ軽いクリーニングを適用
- `keyword_clean` と `text_clean` を結合し、`input_text` を作成

---

## 前処理後のデータ

前処理後、以下のデータフレームを作成した。

### train_processed

| カラム名 | 内容 |
|---|---|
| id | 元データのID |
| keyword | 元のキーワード |
| location | 元の位置情報 |
| text | 元の本文 |
| target | 目的変数 |
| keyword_clean | クリーニング後のキーワード |
| text_clean | クリーニング後の本文 |
| input_text | keyword_clean と text_clean を結合した学習用テキスト |

### test_processed

| カラム名 | 内容 |
|---|---|
| id | 元データのID |
| keyword | 元のキーワード |
| location | 元の位置情報 |
| text | 元の本文 |
| keyword_clean | クリーニング後のキーワード |
| text_clean | クリーニング後の本文 |
| input_text | keyword_clean と text_clean を結合した予測用テキスト |

---

## ベースラインモデル

今回のベースラインモデルには、以下を使用した。

- `TfidfVectorizer`
- `LogisticRegression`
- `Pipeline`

モデル構築では、`train_processed["input_text"]` を説明変数、`train_processed["target"]` を目的変数として使用した。

### データ分割

学習データと検証データは、以下の条件で分割した。

| 項目 | 設定 |
|---|---|
| test_size | 0.2 |
| stratify | y |
| random_state | 42 |

`stratify=y` を指定することで、学習データと検証データで target の分布が大きく崩れないようにした。

---

## TF-IDF の設定

`TfidfVectorizer` では、word ベースの TF-IDF を使用した。

| パラメータ | 設定 |
|---|---|
| ngram_range | (1, 2) |
| min_df | 2 |
| max_df | 0.95 |
| sublinear_tf | True |

### 設定意図

`ngram_range=(1, 2)` により、単語単体だけでなく、2語の並びも特徴量として扱う。

`min_df=2` により、1回しか出現しない極端にレアな語を除外する。

`max_df=0.95` により、ほぼ全体に出現するような情報量の少ない語を除外する。

`sublinear_tf=True` により、単語頻度の影響を対数スケールに変換し、頻出語の影響が強くなりすぎることを抑える。

---

## Logistic Regression の設定

`LogisticRegression` では、以下の設定を使用した。

| パラメータ | 設定 |
|---|---|
| max_iter | 1000 |
| C | 1.0 |
| random_state | 42 |

### 設定意図

`max_iter=1000` にすることで、学習が収束しやすいようにした。

`C=1.0` は正則化の標準的な設定であり、まずはベースラインとして採用した。

`random_state=42` により、実験の再現性を確保した。

---

## 評価指標

検証データに対して、以下の指標を確認した。

- accuracy
- f1_score
- classification_report
- confusion_matrix

今回のタスクでは二値分類であり、クラスごとの precision / recall / f1-score も確認する必要があるため、`classification_report` を表示した。

また、どちらのクラスをどのように間違えているかを見るために、`confusion_matrix` も確認した。

---

## 提出ファイル

学習済みモデルを使って `test_processed["input_text"]` に対して予測を行い、Kaggle 提出用 CSV を作成した。

提出ファイル名は以下。

`submission_tfidf_logreg.csv`

提出ファイルの形式は以下。

| カラム名 | 内容 |
|---|---|
| id | test.csv の id |
| target | 予測ラベル |

CSV 保存時には、Kaggle の提出形式を崩さないように `index=False` を指定した。

### 作成された提出ファイルの形状

`submission_tfidf_logreg.csv` の shape は以下。

| rows | columns |
|---:|---:|
| 3263 | 2 |

### 予測ラベルの分布

今回のベースラインモデルによる test データへの予測分布は以下。

| target | proportion |
|---|---:|
| 0 | 約 0.651 |
| 1 | 約 0.349 |

---

## 現時点での到達点

この時点で、以下の流れを一通り構築できた。

1. Google Drive から train.csv / test.csv を読み込む
2. データの中身、欠損値、カラム、目的変数の分布を確認する
3. TF-IDF 向けの軽いテキスト前処理を行う
4. `input_text` を作成する
5. TF-IDF + LogisticRegression の Pipeline を作成する
6. train / valid に分割して検証する
7. accuracy / f1_score / classification_report / confusion_matrix を確認する
8. test データに対して予測する
9. Kaggle 提出用 CSV を作成する

今回の目的である「前処理コード」と「ベースライン学習コード」の分離は達成できた。

---

## 今回の学び

### 1. NLPタスクでは、まず軽い前処理で十分にベースラインを作れる

最初から Transformer や複雑な特徴量エンジニアリングに進むのではなく、TF-IDF + LogisticRegression でも十分に比較基準となるモデルを作れる。

特に、Kaggle の NLP 二値分類では、まず以下のような構成を作ることが重要だと分かった。

- 入力テキストの確認
- 欠損値処理
- 軽いクリーニング
- TF-IDF
- 線形モデル
- 検証スコア確認
- 提出ファイル作成

### 2. 前処理をやりすぎないことも重要

TF-IDF では、単語そのものが特徴量になる。

そのため、stopwords 除去、stemming、lemmatization を最初から行うと、分類に効く情報を失う可能性がある。

今回は、以下のような軽い処理に留めた。

- 小文字化
- URL置換
- メンション置換
- ハッシュタグ記号の削除
- 空白整理

この方針により、元テキストの情報をできるだけ残しながら、ノイズを軽く減らすことができた。

### 3. keyword は使う価値がある

`keyword` は欠損があるものの、投稿に関連する重要語として使える可能性がある。

そのため、欠損値を空文字で補完した上で、`text` と結合して `input_text` を作成した。

一方で、`location` は欠損が多く、表記ゆれやノイズも多いと考えられるため、最初のベースラインでは使用しなかった。

### 4. Pipeline を使うと実験管理がしやすい

`TfidfVectorizer` と `LogisticRegression` を `Pipeline` でまとめることで、前処理から学習までの流れを一体化できた。

これにより、今後以下のような改善を行う際にも管理しやすくなる。

- TF-IDF のパラメータ変更
- LogisticRegression の C 調整
- class_weight の追加
- char-level TF-IDF の追加
- FeatureUnion による特徴量結合

### 5. Kaggle提出用CSVの作成まで一気通貫で確認できた

モデルを作るだけでなく、最終的に以下の形式で提出CSVを作成できた。

| id | target |
|---|---|

Kaggle では、モデルの検証だけでなく、提出ファイルの形式が正しいことも重要である。

今回、`submission.head()` と `submission.shape` を確認したことで、提出形式の崩れがないことを確認できた。

---

## 今後の改善候補

### 1. LogisticRegression の C を調整する

現在は `C=1.0` を使用している。

今後は以下のような値を試す。

- C=0.3
- C=0.5
- C=1.0
- C=2.0
- C=3.0
- C=5.0

`C` は正則化の強さに関わるパラメータである。

小さいほど正則化が強くなり、大きいほど正則化が弱くなる。

### 2. class_weight="balanced" を試す

target の分布に偏りがある場合、`class_weight="balanced"` が有効な可能性がある。

特に、片方のクラスの recall が低い場合には試す価値がある。

### 3. char-level TF-IDF を試す

word-level TF-IDF だけでなく、文字 n-gram ベースの TF-IDF も有効な可能性がある。

候補設定は以下。

- analyzer="char_wb"
- ngram_range=(3, 5)

char-level TF-IDF は、誤字、表記ゆれ、未知語、短文に対して強い可能性がある。

### 4. word TF-IDF と char TF-IDF を結合する

`FeatureUnion` を用いて、word-level TF-IDF と char-level TF-IDF を結合する。

これにより、単語単位の意味情報と、文字単位の表記パターンの両方を使える。

### 5. threshold 調整を行う

現在は `model.predict()` によるデフォルトの閾値で予測している。

今後は `predict_proba()` を使い、閾値を調整することで F1 score を改善できる可能性がある。

例：

- threshold=0.35
- threshold=0.40
- threshold=0.45
- threshold=0.50
- threshold=0.55

### 6. 前処理の改善

現在はかなり軽い前処理にしている。

今後は、スコアを見ながら以下を検討する。

- 記号の扱いを調整する
- 数字を残すか置換するか検討する
- 絵文字の扱いを検討する
- 短縮表現の正規化を検討する
- URLやメンションを削除するパターンも試す

ただし、前処理を強くしすぎると情報が失われる可能性があるため、必ず検証スコアを見ながら判断する。

---

## 次にやること

次は、今回のベースラインを基準として、以下の順番で改善を進める。

1. 現在の validation score と Kaggle Public LB を記録する
2. LogisticRegression の C を調整する
3. `class_weight="balanced"` を試す
4. char-level TF-IDF を試す
5. word TF-IDF + char TF-IDF の結合を試す
6. threshold 調整を行う
7. 改善結果を README.md に追記する

---

## 現時点の結論

今回、Kaggle NLP 二値分類タスクに対して、TF-IDF + LogisticRegression による最初のベースラインを構築できた。

このベースラインは、以下の点で今後の改善の土台になる。

- シンプルで理解しやすい
- 実行が速い
- 再現性がある
- 前処理と学習コードが分離されている
- Kaggle 提出用 CSV まで作成できる
- 改善実験の比較基準として使える

今後は、このベースラインを崩さずに、パラメータ調整や特徴量追加によってスコア改善を狙う。
