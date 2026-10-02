# Halo — 10 秒產品影片

參考 Apple 宣傳片的風格：開場是黑底大字，接著燈光亮起，鏡頭在鏡片上近拍，高光從鏡面滑過；然後一鏡到底拉遠成主視覺，最後是品名收尾。

成片：`out/halo_film.mp4`（1920×1080，30fps，10 秒，含程式合成的配樂）

| 時間 | 鏡頭 |
|---|---|
| 0–2s | 黑底，標語「Lighter than light.」由模糊淡入，字距逐漸收緊 |
| 2–4.5s | 燈光從黑暗中亮起，鏡頭在右鏡片上近拍並緩慢推移，高光掃過鏡面 |
| 4.5–6.8s | 一鏡到底拉遠，旋轉回正，停在主視覺構圖 |
| 6.8–8s | 產品輕微「呼吸」浮動，第二道高光掃過，下方出現規格文案 |
| 8–10s | 產品上移縮小，「Halo」、中文標語與副標依序出現，最後淡出 |

## 重新渲染

```bash
pip install pillow numpy   # 另需 ffmpeg
python3 render.py --preview
```

文案、顏色和節奏都在 `render.py` 開頭的常數裡，可以直接改。
字型：Inter Display、Noto Sans TC（皆為 SIL Open Font License）。
