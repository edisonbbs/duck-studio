# XgoDuck Studio

本機訓練版本管理網站：配方比較、訓練控制、摘要曲線、checkpoint、ONNX 匯出及成果封存。Play 與 TensorBoard 使用獨立分頁。實機控制與校準由實鴨原有網站處理。

## 運行

此專案需要既有 xgoduck_rl 訓練環境，兩個資料夾需並列。於 Linux／WSL 執行：

```sh
cd xgoduck_studio
../xgoduck_rl/.venv/bin/python server.py
```

開啟 http://127.0.0.1:8092 。Windows 的 Start Studio.cmd 為目前電腦的 WSL 路徑範例，其他電腦需按實際路徑修改。

GitHub 儲存程式與 recipes/ 參數紀錄；不提供 GPU 訓練主機。GitHub Pages 無法執行這個 Python 後端。

## 資料與版本

recipes/ 保存可追蹤的參數摘要及來源摘要雜湊，並不包含可獨立重現訓練所需的完整源碼。完整版本快照、模型、日誌、成果及 runtime 設定留在本機，由 .gitignore 白名單排除。新建本機版本後，參數摘要尚未自動同步至 recipes/。

首次運行匯入鄰近 xgoduck_rl 作為 v001；不是自動還原此 Git 儲存庫記錄的歷史訓練。版本使用共用 venv；原始碼快照不等於完整套件環境隔離。根部高度調整仍需訓練驗證。

## 檢查與限制

```sh
../xgoduck_rl/.venv/bin/python test_studio.py
```

本機服務僅監聽 loopback。目前 TensorBoard 須另行啟動。服務重啟後不能自動恢復先前子程序的控制記錄。請勿為更新網站而中斷正在進行的訓練。

## GitHub Pages 查看版

網站來源為 `docs/`。在 GitHub 儲存庫 Settings → Pages 選擇 Deploy from a branch，選擇發佈分支的 `/docs`。部署後可跨地區查看快照，不提供遠端訓練操作。

更新資料：在本機執行 `../xgoduck_rl/.venv/bin/python export_pages.py`，審閱 `docs/data.json` 後提交並推送。此流程沒有自動上傳，也沒有背景定時同步。

匯出白名單包含版本 ID/名稱/父版本/數值參數、訓練目錄名稱、曲線及 checkpoint 檔名。排除備註、絕對路徑、金鑰、模型及原始事件檔。GitHub Pages 的可見性須獨立確認，私人儲存庫不保證網站私有。
