# Smart Burst Selection Pipeline

## Problem Statement
我們該如何從混合了多種連拍場景的資料夾中，精確切割出不同的連拍事件，並根據拍攝主體（人物 vs. 風景/動態）自動適用不同的評分規則，挑選出該組連拍中的最佳照片？

## Recommended Direction
**智慧連拍分組 + 規則引擎 (The Smart Pipeline)**
徹底翻新目前「全資料夾只挑一張」的錯誤設計。程式將首先利用「EXIF 拍攝時間差距 (< 2秒)」與「感知雜湊 (pHash) 距離」來將照片正確切割為多組獨立的連拍 (Burst Events)。
在評分階段，引入「場景感知 (Context-Aware)」機制：若偵測到人臉，則利用 YuNet 的臉部特徵點 (Landmarks) 評估眼睛是否睜開及表情狀態；若無人臉，則切換至風景模式，純粹以全局銳利度與三分法則/畫面平衡度作為評估標準。

## Key Assumptions to Validate
- [ ] **EXIF 的可靠性**：相機寫入的 EXIF 時間戳記精確度（秒級）足以用來區分不同的連拍事件（搭配 pHash 作為雙重確認）。
- [ ] **特徵點準確度**：YuNet 輕量級模型輸出的 5 個五官特徵點，足夠用來寫一個簡單的 heuristic (啟發式) 演算法來判斷「是否眨眼」。
- [ ] **效能衝擊**：讀取 EXIF 與增加特徵點計算，依然能維持在 100 張相片 5 秒內處理完畢的效能目標。

## MVP Scope
- **IN**：利用 `Pillow` 提取 EXIF 拍攝時間，實作 `Time + pHash` 的雙重分組邏輯。
- **IN**：修改 `process_folder` 迴圈，確保是「每組連拍挑選一張最佳」，而非全資料夾挑一張。
- **IN**：擴充 `compute_composition`，當偵測到人臉時，抓取 Landmarks 計算眼睛張開程度；若無人臉，改用畫面梯度分佈評估重心。
- **IN**：在 `defects.csv` 報告中加入「所屬群組 (Burst ID)」欄位。

## Not Doing (and Why)
- **不導入 AI 美學模型 (Deep Learning Aesthetics)**：會拖垮處理速度，且造成 CLI 檔案過於肥大，違反輕量化跨平台的初衷。
- **不作影像合成 (Image Merging/Stacking)**：為了修復眨眼而把多張臉合成，極易產生鬼影破綻，反而增加使用者的困擾。
- **不作高階動作識別 (Advanced Action Recognition)**：不嘗試去理解「球是否剛好碰到球棒」這類語意，單純依賴「邊緣銳利度 (無動態模糊)」來捕捉動作瞬間。

## Open Questions
- (已解決) 根據 `raw_test` 內的範例檔案時間戳記分析，連拍間隔（< 1 秒）與不同組的間隔（約 8 秒）界線明確。因此，連拍時間判定間距將**預設為 2 秒**，並開放讓使用者在 `config.json` 中自訂 (`burst_time_threshold`)。
