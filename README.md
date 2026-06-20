# Kaggle NLP Binary Classification: BERT Baseline

## 概要

本プロジェクトでは、Kaggle の NLP 二値分類タスクに対して、まず `TF-IDF + LogisticRegression` によるベースラインを構築し、その後、2つ目の検証モデルとして `bert-base-uncased` を用いた fine-tuning を行った。

結果として、`bert-base-uncased` を使用したモデルでは Kaggle Public LB において **0.83152** までスコアが上昇した。

この結果から、単語の表層的な出現頻度を扱う TF-IDF よりも、文脈や単語同士の意味的な関係を捉えられる BERT 系モデルの方が、このタスクに対して高い性能を発揮する可能性が示された。

---

## タスク概要

### 目的

`text` の内容から `target` を予測する二値分類タスク。

### 使用データ

- `train.csv`
- `test.csv`

### train.csv のカラム

| カラム名 | 内容 |
|---|---|
| id | 各データのID |
| keyword | テキストに関連するキーワード |
| location | 投稿者の位置情報 |
| text | 投稿本文 |
| target | 目的変数 |

### test.csv のカラム

| カラム名 | 内容 |
|---|---|
| id | 各データのID |
| keyword | テキストに関連するキーワード |
| location | 投稿者の位置情報 |
| text | 投稿本文 |

---

## これまでの検証モデル

### 1. TF-IDF + LogisticRegression

最初のベースラインとして、`TfidfVectorizer` と `LogisticRegression` を組み合わせたモデルを構築した。

#### 特徴

- 実装が簡単
- 学習が高速
- 解釈しやすい
- ベースラインとして優秀
- 小規模データでも安定しやすい

#### 使用した特徴量

`keyword_clean` と `text_clean` を結合した `input_text` を使用。

#### TF-IDF の主な設定

| パラメータ | 設定 |
|---|---|
| ngram_range | (1, 2) |
| min_df | 2 |
| max_df | 0.95 |
| sublinear_tf | True |

#### LogisticRegression の主な設定

| パラメータ | 設定 |
|---|---|
| max_iter | 1000 |
| C | 1.0 |
| random_state | 42 |

#### 課題

TF-IDF は単語の出現頻度をもとに特徴量を作るため、文脈や単語同士の意味的な近さを直接扱うことができない。

たとえば、意味的には近い単語であっても、表記が異なれば別の特徴量として扱われる。また、否定表現や文全体のニュアンス、単語の並びによる意味変化を捉えることが難しい。

そのため、一定のスコアまでは出やすい一方で、そこから先は頭打ちになりやすいと考えられる。

---

### 2. bert-base-uncased

2つ目の検証モデルとして、`bert-base-uncased` を使用した。

#### 結果

| モデル | Public LB |
|---|---:|
| bert-base-uncased | 0.83152 |

`TF-IDF + LogisticRegression` と比較して、BERT 系モデルを用いることでスコアが上昇した。

---

## BERT 用の前処理方針

BERT では、TF-IDF のように人間側で強く単語を整えるのではなく、できるだけ元の文章情報を残し、Tokenizer と事前学習済みモデルに任せる方針を取った。

### 実施した前処理

- `keyword` の欠損値を空文字で補完
- `text` の欠損値を空文字で補完
- HTML エスケープを戻す
- URL エンコードを戻す
- URL を `url` に置換
- メンションを `user` に置換
- 連続する空白を1つにまとめる
- 前後の空白を削除する
- `location` は今回も使用しない

### 実施しなかった前処理

- stopwords 除去
- stemming
- lemmatization
- 強い記号削除
- 明示的な小文字化

`bert-base-uncased` は tokenizer 側で小文字化を行うため、前処理段階では明示的な小文字化を行わなかった。

---

## BERT の入力形式

今回の BERT モデルでは、`keyword` と `text` を sentence pair として tokenizer に渡した。

概念的には、以下のような入力になる。

| 入力要素 | 内容 |
|---|---|
| sentence 1 | keyword |
| sentence 2 | text |

BERT 内部では、概ね以下のような形式で扱われる。

`[CLS] keyword [SEP] text [SEP]`

これにより、単に `keyword` と `text` を文字列結合するのではなく、BERT の sentence pair 入力として自然に扱うことができる。

---

## 学習設定

### 使用モデル

| 項目 | 設定 |
|---|---|
| model_name | bert-base-uncased |
| num_labels | 2 |
| max_length | 128 |
| optimizer | Trainer のデフォルト |
| learning_rate | 2e-5 |
| weight_decay | 0.01 |
| epochs | 3 |
| batch_size | 16 |
| eval_batch_size | 32 |
| random_state | 42 |

### 検証方法

学習データを `train_test_split` で分割した。

| 項目 | 設定 |
|---|---|
| test_size | 0.2 |
| stratify | target |
| random_state | 42 |

評価指標として以下を確認した。

- accuracy
- f1
- classification_report
- confusion_matrix

---

## なぜ BERT でスコアが上がったと考えられるか

### 1. TF-IDF は単語の出現頻度ベース

TF-IDF は、ある単語が文書内でどれだけ重要かを、出現頻度と文書全体での希少性から計算する。

そのため、シンプルで強力ではあるが、基本的には単語を独立した特徴量として扱う。

つまり、以下のような情報は捉えにくい。

- 単語同士の意味的な近さ
- 文脈による意味の変化
- 否定表現
- 皮肉や比喩
- 語順によるニュアンス
- 文全体の意味

### 2. BERT は文脈を考慮できる

BERT は Transformer Encoder をベースとした事前学習済み言語モデルである。

Transformer の Self-Attention 機構により、文中の各単語が他の単語とどのように関係しているかを考慮しながら表現を作ることができる。

そのため、BERT は単語を単体で見るのではなく、文脈の中で意味を捉えることができる。

たとえば、同じ単語であっても、周囲の単語によって意味が変わる場合がある。BERT はこのような文脈依存の意味表現を扱えるため、TF-IDF よりも高い精度を出しやすいと考えられる。

### 3. 事前学習済みモデルの知識を利用できる

`bert-base-uncased` は、大規模なテキストコーパスで事前学習されている。

そのため、今回の Kaggle データだけから学習するのではなく、事前学習で獲得した一般的な言語理解能力を利用できる。

小規模から中規模の NLP タスクでは、この事前学習済みの知識が大きな効果を持つ。

今回のタスクでも、BERT が事前学習によって獲得した文脈理解能力が、分類精度の向上につながったと考えられる。

---

## 現時点の仮説

今回、`bert-base-uncased` によってスコアが **0.83152** まで上昇した。

この結果から、以下の仮説が立てられる。

### 仮説1

このタスクでは、単語の出現頻度だけでなく、文脈や単語同士の意味的な関係が重要である。

そのため、TF-IDF よりも Transformer ベースのモデルの方が高い精度を出しやすい。

### 仮説2

BERT の Self-Attention 機構により、文中の単語間の関係を考慮できることが、今回のスコア向上に寄与している。

### 仮説3

`bert-base-uncased` よりも事前学習方法が改善されている RoBERTa を使用すれば、さらに高い精度が出る可能性がある。

---

## 次に検証したいモデル: RoBERTa

次の検証候補として、`roberta-base` を試す。

RoBERTa は BERT をベースにしつつ、事前学習の方法を改善したモデルである。

### RoBERTa に期待する理由

- BERT よりも強力な事前学習設定を持つ
- 動的マスキングなどにより、より汎用的な言語表現を獲得している
- NSP を使わない事前学習により、より自然な文脈表現を学習している
- NLP 分類タスクで BERT より高い性能を出すことが多い

そのため、今回のような短文ベースの二値分類タスクでも、`bert-base-uncased` より高いスコアが出る可能性がある。

---

## RoBERTa 検証時の注意点

RoBERTa では BERT と tokenizer の仕様が異なる。

そのため、以下の点に注意する。

### 1. token_type_ids を使わない

BERT では sentence pair に対して `token_type_ids` が使われる。

一方、RoBERTa では通常 `token_type_ids` を使用しない。

そのため、Dataset 側では `encoding` に `token_type_ids` が存在する場合のみ渡す実装にしておくと安全である。

### 2. 前処理を強くしすぎない

RoBERTa でも、BERT と同様に過剰な前処理は避ける。

特に以下は行わない。

- stopwords 除去
- stemming
- lemmatization
- 強い記号削除

### 3. max_length はまず 128 でよい

今回のデータは短文中心であるため、最初は `max_length=128` で十分だと考えられる。

必要に応じて、以下を比較する。

- max_length=128
- max_length=160
- max_length=192

### 4. 学習率は BERT と同じく 2e-5 から始める

まずは BERT と条件を揃えて比較するため、以下の設定で開始する。

| 項目 | 設定 |
|---|---|
| learning_rate | 2e-5 |
| epochs | 3 |
| batch_size | 16 |
| weight_decay | 0.01 |

その後、必要に応じて以下を試す。

- learning_rate=1e-5
- learning_rate=2e-5
- learning_rate=3e-5
- epochs=2
- epochs=3
- epochs=4

---

## 今後の実験方針

今後は、以下の順番で検証を進める。

### 1. bert-base-uncased の結果を基準として記録する

現在のベストスコア。

| モデル | Public LB |
|---|---:|
| bert-base-uncased | 0.83152 |

このスコアを次のモデルの比較基準とする。

### 2. roberta-base を検証する

BERT と同じデータ分割、同じ評価指標で検証する。

比較する観点は以下。

- validation accuracy
- validation f1
- Public LB
- 予測ラベルの分布
- confusion matrix

### 3. ハイパーパラメータを調整する

RoBERTa で改善が見られた場合、以下を調整する。

- learning_rate
- epochs
- batch_size
- max_length
- weight_decay
- threshold

### 4. threshold tuning を行う

現在は `argmax` によって予測ラベルを決定している。

今後は `predict_proba` 相当の確率値を使い、閾値を調整することで F1 score の改善を狙う。

候補は以下。

| threshold |
|---:|
| 0.35 |
| 0.40 |
| 0.45 |
| 0.50 |
| 0.55 |
| 0.60 |

### 5. アンサンブルを検討する

BERT と RoBERTa の予測傾向が異なる場合、アンサンブルによって性能が上がる可能性がある。

候補は以下。

- BERT + RoBERTa の確率平均
- BERT + TF-IDF LogisticRegression の確率平均
- RoBERTa + TF-IDF LogisticRegression の確率平均
- 複数 seed の平均

---

## 現時点の結論

`TF-IDF + LogisticRegression` は高速で強いベースラインとして有効だった。

しかし、今回 `bert-base-uncased` を使用したことで Public LB が **0.83152** まで上昇したことから、このタスクでは文脈理解や単語同士の意味的な関係が重要である可能性が高い。

BERT は Transformer の Self-Attention 機構によって文中の単語間の関係を捉えることができ、さらに大規模コーパスでの事前学習によって一般的な言語理解能力を持っている。

そのため、単語の出現頻度を中心に扱う TF-IDF よりも高い精度を出せたと考えられる。

次の検証では、BERT よりも事前学習が改善されている `roberta-base` を使用し、さらなるスコア向上を狙う。
