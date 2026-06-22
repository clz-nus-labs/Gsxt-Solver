# 语义排序词库目录

`dynamic_mixed_infer.py` 会自动递归读取本目录下的 `*.txt`、`*.dict`、`*.csv`、`*.tsv` 文件。

推荐放置方式：

- `manual/`：手工补充的短词、常用词、容易误排的词。
- `thuocl/`：THUOCL 词表文件，可直接放原始 `.txt`。
- `generated/`：由 `Scripts/Gsxt/tools/build_semantic_lexicon.py` 从大语料抽取出的轻量词表。
- `jieba/`：随项目加载的通用中文词频词典，来源于
  [fxsjy/jieba](https://github.com/fxsjy/jieba/blob/master/jieba/dict.txt)，用于联合解码的词频先验。
- `raw_corpus/`：原始大语料或 zip 包。推理脚本不会读取这里，需先转换到 `generated/`。

构建大语料词表示例：

```powershell
& "$env:CONDA_PREFIX\python.exe" .\Scripts\Gsxt\tools\build_semantic_lexicon.py --source .\Scripts\Gsxt\data\semantic_lexicons\raw_corpus --output .\Scripts\Gsxt\data\semantic_lexicons\generated\large_corpus_phrases.txt --min-len 2 --max-len 5 --min-count 2 --top-k 200000 --use-jieba
```
