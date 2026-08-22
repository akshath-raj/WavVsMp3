# Bilingual abstract

The `academic-paper` pipeline produces abstracts in English and 繁體中文. The
English version is the one set in `main.tex`; the Chinese version is kept here
rather than in the manuscript, since an Interspeech submission is
English-only.

---

## English

Speech emotion recognition (SER) is validated on uncompressed audio and
deployed on coded audio. We measure what that substitution costs and find the
answer is not a property of the codec. Fifteen classifiers — fourteen
scikit-learn models and a neural network — are trained on 436 named acoustic
descriptors from CREMA-D under a speaker-independent split, then served MP3 and
MP4/AAC at 64 kbps. Under MP3 the outcome is categorical rather than graded:
every model that thresholds a feature value stays within 1.4 points of its
clean balanced accuracy, while every model that multiplies one loses 11.7 to
38.3 points. The two families do not overlap. Because the features are named
the mechanism is legible: perceptual bit allocation abandons the top octave,
and one descriptor of that octave arrives 38.6 training standard deviations out
of range. Masking that single column recovers 97.9 % of an RBF-SVM's loss and
92.3 % of the network's at no cost on clean audio, so the failure is covariate
shift on an identifiable channel rather than lost emotional information.
Attribution stability follows the same family split, and we retain two negative
results constraining how such stability should be used.

**Index terms:** speech emotion recognition, perceptual audio coding, covariate
shift, explainable AI, model robustness

---

## 繁體中文

語音情緒辨識系統多以未壓縮音訊驗證，卻在實際部署時處理經過編碼的音訊。本研究量測此
一替換的代價，並發現代價的大小並非取決於編碼器本身。我們以 CREMA-D 語料庫萃取 436
個具名聲學特徵，在語者獨立切分下訓練十五個分類器（十四個 scikit-learn 模型與一個類
神經網路），再分別以 64 kbps 的 MP3 與 MP4/AAC 進行測試。在 MP3 條件下，結果呈現類別
性而非漸進性的差異：凡是以門檻方式使用特徵值的模型，其平衡準確率變動不超過 1.4 個百
分點；凡是以乘法方式使用特徵值的模型，則損失 11.7 至 38.3 個百分點，兩類模型之間毫無
重疊。由於特徵具有明確的聲學意義，其機制得以辨識：感知位元配置捨棄了最高八度音程，而
描述該頻段的單一特徵在推論時偏離訓練分布達 38.6 個標準差。僅遮蔽該單一特徵，即可回復
RBF-SVM 損失的 97.9 % 與類神經網路損失的 92.3 %，且對未壓縮音訊的效能毫無損害；可見
問題在於可辨識通道上的共變量偏移，而非情緒資訊的流失。歸因穩定性亦依循相同的模型族群
分野，我們並保留兩項負面結果，用以界定此類穩定性指標的適用範圍。

**關鍵詞：** 語音情緒辨識、感知音訊編碼、共變量偏移、可解釋人工智慧、模型穩健性
